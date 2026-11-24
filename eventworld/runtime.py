"""The released P40530 runtime, using frozen EventVGGT readouts."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
from safetensors.torch import load_file

from .assets import EVENTVGGT_SHA256, HEIGHT, P40530_SHA256, WIDTH, sha256
from .predictor import ForecastCore

POSE_SHA256 = "35e071e82e036501eaf223b998ed541f702f99d76631805e35061589343cca37"
PATCH_HW = (18, 36)


def _require_identity(path: Path, expected: str) -> None:
    if sha256(path) != expected:
        raise ValueError(f"Unexpected checkpoint SHA-256: {path}")


def _upstream_state(path: Path, checksum: str) -> dict:
    # The official archives include OmegaConf metadata. Only deserialize the
    # exact upstream files identified by the release's SHA-256 values.
    _require_identity(path, checksum)
    saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    tensors = saved.get("model", saved)
    if not tensors or not all(isinstance(name, str) and isinstance(value, torch.Tensor)
                              for name, value in tensors.items()):
        raise ValueError(f"Expected an upstream tensor state dictionary: {path}")
    normalized = {name.removeprefix("module."): tensor for name, tensor in tensors.items()}
    if len(normalized) != len(tensors):
        raise ValueError(f"Duplicate upstream checkpoint keys: {path}")
    return normalized


def _load_eventvggt(checkpoint: Path):
    from vggt.models.vggt_lora_dino import create_vggt_with_lora_dino, create_vggt_with_split_qkv

    base = create_vggt_with_split_qkv(
        img_size=518, patch_size=14, embed_dim=1024, enable_camera=False,
        enable_depth=True, enable_point=False, enable_track=False,
    )
    wrapped = create_vggt_with_lora_dino(base, lora_r=16, lora_alpha=32)
    state = _upstream_state(checkpoint, EVENTVGGT_SHA256)
    result = wrapped.load_state_dict(state, strict=False)
    expected = wrapped.state_dict()
    matched = sum(value.numel() for key, value in expected.items() if key in state)
    coverage = matched / sum(value.numel() for value in expected.values())
    critical = [key for key in result.missing_keys if "lora_" in key or "depth_head" in key]
    if coverage < 0.99 or critical:
        raise RuntimeError(f"EventVGGT state coverage {coverage:.3%}, critical missing tensors: {critical}")
    wrapped.requires_grad_(False).eval().cuda()
    return wrapped.base_model.model


def _load_camera_head(checkpoint: Path):
    from vggt.heads.camera_head import CameraHead

    head = CameraHead(dim_in=2048)
    prefix = "base_model.model.camera_head."
    state = {key.removeprefix(prefix): value for key, value in _upstream_state(checkpoint, POSE_SHA256).items()
             if key.startswith(prefix)}
    head.load_state_dict(state, strict=True)
    return head.requires_grad_(False).eval().cuda()


class EventWorld:
    def __init__(self, weights: Path, eventvggt_checkpoint: Path, camera_checkpoint: Path | None = None):
        if (not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8
                or not torch.cuda.is_bf16_supported()):
            raise RuntimeError("EventWorld requires an NVIDIA CUDA GPU with native bfloat16 support")
        _require_identity(weights, P40530_SHA256)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        # The upstream snapshots use top-level vggt imports.
        vendor = str(Path(__file__).resolve().parent / "_vendor")
        if vendor not in sys.path:
            sys.path.insert(0, vendor)
        from ._vendor.author_fm import FMConfig, Flowmatching

        config = FMConfig(in_dim=1024, model_dim=1024, depth=8, n_heads=16, mlp_ratio=4.0,
                          attn_drop=0.0, proj_drop=0.0, n_max=20000, t_frames=2)
        base = Flowmatching(config).cuda().float()
        base.timesteps_per_stage = base.timesteps_per_stage.float()
        base.sigmas_per_stage = base.sigmas_per_stage.float()
        self.predictor = ForecastCore(base).cuda().float()
        self.predictor.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
        self.predictor.requires_grad_(False).eval()
        self.event = _load_eventvggt(eventvggt_checkpoint)
        self.camera = _load_camera_head(camera_checkpoint) if camera_checkpoint is not None else None

    def _readout(self, tokens: torch.Tensor, include_camera: bool = False):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            features, patch_start = self.event.aggregator.part2([tokens], patch_hw=PATCH_HW)
        indices = self.event.depth_head.intermediate_layer_idx
        slots = [None] * (max(indices) + 1)
        features = [feature.float() for feature in features]
        for index, feature in zip(indices, features, strict=True):
            slots[index] = feature
        # The DPT head consumes image dimensions here. No target image is needed.
        shape = tokens.new_zeros((tokens.shape[0], tokens.shape[1], 3, HEIGHT, WIDTH))
        with torch.autocast("cuda", enabled=False):
            depth, _ = self.event.depth_head(slots, images=shape, patch_start_idx=patch_start)
            extrinsics = None
            if include_camera:
                from vggt.utils.pose_enc import pose_encoding_to_extri_intri
                poses = self.camera(features)[-1]
                ext, _ = pose_encoding_to_extri_intri(poses, image_size_hw=(HEIGHT, WIDTH), build_intrinsics=False)
                extrinsics = torch.eye(4, device=tokens.device).expand(1, tokens.shape[1], 4, 4).clone()
                extrinsics[:, :, :3] = ext.float()
        return depth[..., 0].float(), extrinsics

    @torch.inference_mode()
    def forecast(self, images: np.ndarray, motion: np.ndarray, horizons: tuple[int, ...], seed: int) -> dict:
        image = torch.from_numpy(images).permute(0, 3, 1, 2).unsqueeze(0).cuda()
        motion_features = torch.from_numpy(motion).cuda().float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            history = self.event.aggregator.part1(image)[0][0].float()
        condition = self.predictor.condition(history, motion_features)
        generator = torch.Generator(device=history.device).manual_seed(seed)
        noise = torch.randn((1, 1, 653, 1024), generator=generator, device=history.device, dtype=torch.float32)
        current, _ = self._readout(history)
        forecasts, poses = [], []
        for horizon in horizons:
            future = self.predictor.endpoint(noise, condition, horizon)
            depth, extrinsics = self._readout(torch.cat((history, future), dim=1), self.camera is not None)
            forecasts.append(depth[0, -1].cpu().numpy())
            if extrinsics is not None:
                poses.append(extrinsics[0, -2:].cpu().numpy())
        result = {
            "horizons_ms": np.asarray(horizons, dtype=np.int32),
            "depth": np.stack(forecasts),
            "current_depth": current[0, -1].cpu().numpy(),
        }
        if not all(np.isfinite(result[key]).all() and (result[key] > 0).all()
                   for key in ("depth", "current_depth")):
            raise FloatingPointError("Decoder produced a non-finite or non-positive depth")
        if poses:
            result["world_to_camera"] = np.stack(poses)
            if not np.isfinite(result["world_to_camera"]).all():
                raise FloatingPointError("Camera head produced non-finite extrinsics")
        return result


def backproject(depth: np.ndarray, intrinsics: tuple[float, float, float, float]) -> np.ndarray:
    """Convert relative depth to XYZ in the camera frame at each horizon."""
    fx, fy, cx, cy = intrinsics
    y, x = np.indices(depth.shape[-2:], dtype=np.float32)
    return np.stack(((x - cx) / fx * depth, (y - cy) / fy * depth, depth), axis=-1)

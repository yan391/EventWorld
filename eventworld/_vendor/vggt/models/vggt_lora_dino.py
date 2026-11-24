# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
VGGT LoRA for DINO ViT backbone only - Q and V projections only.

This module applies LoRA fine-tuning ONLY to the Q (query) and V (value) projections 
in the patch_embed (DINO ViT) within the Aggregator. It explicitly EXCLUDES:
- K (key) projections in DINO ViT
- frame_blocks Q/K/V
- global_blocks Q/K/V

This selective approach fine-tunes the query and value transformations while keeping
the key projections frozen, which can be beneficial for preserving learned attention
patterns while adapting to new visual domains (e.g., RGB to Event images).

Requirements:
- The model must be created with split_qkv=True in the Aggregator
- Pretrained weights will be automatically converted from joint QKV to split Q/K/V
"""

import logging
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, PeftModel

from vggt.models.vggt import VGGT

logger = logging.getLogger(__name__)


def create_vggt_with_lora_dino(
    base_model: VGGT,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    target_modules: Optional[list] = None,
) -> PeftModel:
    """
    Create a VGGT model with LoRA adapters applied ONLY to Q and V projections
    in the patch_embed (DINO ViT backbone).
    
    This differs from the standard vggt_lora.py by:
    - Only targeting patch_embed.blocks (DINO ViT)
    - NOT targeting frame_blocks or global_blocks
    
    Args:
        base_model: The base VGGT model (should be initialized with split_qkv=True).
        lora_r: LoRA rank parameter.
        lora_alpha: LoRA alpha parameter for scaling.
        lora_dropout: Dropout probability for LoRA layers.
        target_modules: List of module names to apply LoRA to. If None, defaults to
                       all q_proj and v_proj in patch_embed.blocks only.
    
    Returns:
        PeftModel: VGGT model with LoRA adapters attached to DINO ViT only.
    
    Example:
        >>> base_model = VGGT(img_size=518, patch_size=14, embed_dim=1024)
        >>> model_with_lora = create_vggt_with_lora_dino(base_model, lora_r=8)
    """
    # Default target modules: only q_proj and v_proj in patch_embed.blocks
    if target_modules is None:
        target_modules = []
        
        # Check if patch_embed has blocks (i.e., it's a vision transformer)
        if hasattr(base_model.aggregator.patch_embed, 'blocks'):
            vit_blocks = base_model.aggregator.patch_embed.blocks
            
            # Check first block to verify split_qkv structure
            if len(vit_blocks) > 0:
                first_block = vit_blocks[0]
                if hasattr(first_block, 'attn'):
                    attn = first_block.attn
                    # Check if using split Q/K/V projections
                    uses_split_qkv = (hasattr(attn, 'q_proj') and 
                                     hasattr(attn, 'v_proj') and 
                                     hasattr(attn, 'k_proj') and
                                     attn.q_proj is not None)
                    
                    if uses_split_qkv:
                        # Perfect! Apply LoRA only to q_proj and v_proj, NOT k_proj
                        logger.info(f"Detected split Q/K/V in patch_embed.blocks")
                        logger.info(f"Applying LoRA to q_proj and v_proj ONLY (NOT k_proj) in {len(vit_blocks)} blocks")
                        for i in range(len(vit_blocks)):
                            target_modules.append(f"aggregator.patch_embed.blocks.{i}.attn.q_proj")
                            target_modules.append(f"aggregator.patch_embed.blocks.{i}.attn.v_proj")
                            # Explicitly NOT adding k_proj - this is the key difference!
                    else:
                        logger.error(
                            "patch_embed blocks do not have split Q/K/V projections. "
                            "Please ensure the model is created with split_qkv=True."
                        )
                        raise ValueError(
                            "Model must be initialized with split_qkv=True in the Aggregator "
                            "to apply LoRA selectively to Q and V only. "
                            "Use create_vggt_with_split_qkv() to create the model."
                        )
                else:
                    logger.warning("First block in patch_embed has no 'attn' attribute")
            else:
                logger.warning("patch_embed.blocks is empty")
        else:
            logger.warning(
                "patch_embed does not have 'blocks' attribute. "
                "This might be a convolutional patch_embed without transformer layers."
            )
    
    logger.info(f"Applying LoRA to {len(target_modules)} modules in DINO ViT backbone")
    logger.info(f"LoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    
    if len(target_modules) == 0:
        logger.warning("No target modules found for LoRA. Returning base model unchanged.")
        return base_model
    
    # Create LoRA configuration
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=None,  # Not using a predefined task type
    )
    
    # Apply LoRA to the model
    model_with_lora = get_peft_model(base_model, lora_config)
    
    # Log trainable parameters
    trainable_params, all_params = model_with_lora.get_nb_trainable_parameters()
    logger.info(
        f"Trainable params: {trainable_params:,} || "
        f"All params: {all_params:,} || "
        f"Trainable%: {100 * trainable_params / all_params:.2f}%"
    )
    
    return model_with_lora


def initialize_dual_models(
    model_config: Dict[str, Any],
    lora_config: Dict[str, Any],
    pretrained_weights_path: Optional[str] = None,
) -> tuple[VGGT, PeftModel]:
    """
    Initialize two VGGT models for Event-to-Depth training with LoRA on DINO ViT only:
    1. RGB model (frozen): Used to generate pseudo ground truth depth maps.
    2. Event model (with LoRA on DINO ViT only): Fine-tuned on event images.
    
    This differs from initialize_dual_models() by only applying LoRA to the 
    patch_embed (DINO ViT backbone), leaving frame_blocks and global_blocks unchanged.
    
    Args:
        model_config: Configuration dict for VGGT model initialization.
        lora_config: Configuration dict for LoRA (r, alpha, dropout).
        pretrained_weights_path: Path to pretrained weights to load.
    
    Returns:
        tuple: (model_rgb, model_event)
            - model_rgb: Frozen VGGT model for RGB input
            - model_event: VGGT model with LoRA on DINO ViT only for Event input
    
    Example:
        >>> model_config = {
        ...     "img_size": 518,
        ...     "patch_size": 14,
        ...     "embed_dim": 1024,
        ...     "enable_depth": True,
        ... }
        >>> lora_config = {"lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.1}
        >>> model_rgb, model_event = initialize_dual_models(model_config, lora_config)
    """
    logger.info("Initializing dual VGGT models for Event-to-Depth training (LoRA on DINO ViT only)")
    
    # Initialize RGB model (without split_qkv, standard architecture)
    logger.info("Creating RGB model (frozen)...")
    model_rgb = VGGT(**model_config)
    
    # Load pretrained weights if provided
    if pretrained_weights_path is not None:
        logger.info(f"Loading pretrained weights from {pretrained_weights_path}")
        checkpoint = torch.load(pretrained_weights_path, map_location="cpu")
        model_state = checkpoint.get("model", checkpoint)
        model_rgb.load_state_dict(model_state, strict=False)
    
    # Freeze all parameters in RGB model
    for param in model_rgb.parameters():
        param.requires_grad = False
    model_rgb.eval()  # Set to eval mode
    logger.info("RGB model frozen and set to eval mode")
    
    # Initialize Event model (with split_qkv for LoRA)
    logger.info("Creating Event model (with LoRA on DINO ViT only)...")
    model_config_event = model_config.copy()
    # Need to create aggregator with split_qkv=True
    model_event = create_vggt_with_split_qkv(**model_config)
    
    # Load the same pretrained weights into Event model
    if pretrained_weights_path is not None:
        logger.info("Loading pretrained weights into Event model...")
        # Load weights, converting from joint qkv to split q/k/v if necessary
        load_weights_with_qkv_conversion(model_event, model_state)
    
    # Apply LoRA to Event model (DINO ViT only)
    logger.info("Applying LoRA adapters to Event model (DINO ViT only)...")
    model_event = create_vggt_with_lora_dino(
        model_event,
        lora_r=lora_config.get("lora_r", 8),
        lora_alpha=lora_config.get("lora_alpha", 16),
        lora_dropout=lora_config.get("lora_dropout", 0.1),
    )
    
    logger.info("Dual model initialization complete (LoRA on DINO ViT only)")
    return model_rgb, model_event


def create_vggt_with_split_qkv(**kwargs) -> VGGT:
    """
    Create a VGGT model with split_qkv=True in the aggregator.
    
    This is a helper function that creates a VGGT instance where the aggregator
    uses separate Q, K, V projections instead of a joint QKV projection.
    
    Args:
        **kwargs: Arguments passed to VGGT constructor.
    
    Returns:
        VGGT: Model instance with split QKV projections in aggregator.
    """
    # Create a modified model where aggregator has split_qkv=True
    class VGGTWithSplitQKV(VGGT):
        def __init__(self, **model_kwargs):
            # Extract aggregator config
            img_size = model_kwargs.get("img_size", 518)
            patch_size = model_kwargs.get("patch_size", 14)
            embed_dim = model_kwargs.get("embed_dim", 1024)
            
            # Initialize parent without calling __init__ first
            super(VGGT, self).__init__()
            
            # Import here to avoid circular dependency
            from vggt.models.aggregator import Aggregator
            from vggt.heads.camera_head import CameraHead
            from vggt.heads.dpt_head import DPTHead
            from vggt.heads.track_head import TrackHead
            
            # Create aggregator with split_qkv=True
            self.aggregator = Aggregator(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                split_qkv=True,  # Enable split QKV for LoRA
            )
            
            # Create heads as in original VGGT
            enable_camera = model_kwargs.get("enable_camera", True)
            enable_point = model_kwargs.get("enable_point", True)
            enable_depth = model_kwargs.get("enable_depth", True)
            enable_track = model_kwargs.get("enable_track", True)
            
            self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
            self.point_head = DPTHead(
                dim_in=2 * embed_dim,
                output_dim=4,
                activation="inv_log",
                conf_activation="expp1"
            ) if enable_point else None
            self.depth_head = DPTHead(
                dim_in=2 * embed_dim,
                output_dim=2,
                activation="exp",
                conf_activation="expp1"
            ) if enable_depth else None
            self.track_head = TrackHead(
                dim_in=2 * embed_dim,
                patch_size=patch_size
            ) if enable_track else None
    
    return VGGTWithSplitQKV(**kwargs)


def load_weights_with_qkv_conversion(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    """
    Load weights into a model with split QKV, converting from joint QKV if necessary.
    
    When loading pretrained weights from a model with joint QKV projection into a model
    with split Q/K/V projections, this function splits the joint weights appropriately.
    
    Args:
        model: Target model with split Q/K/V projections.
        state_dict: State dict from a model with joint QKV projection.
    """
    new_state_dict = {}
    
    for name, param in state_dict.items():
        # Check if this is a joint QKV weight that needs to be split
        if ".qkv.weight" in name:
            # Split into Q, K, V
            q_name = name.replace(".qkv.weight", ".q_proj.weight")
            k_name = name.replace(".qkv.weight", ".k_proj.weight")
            v_name = name.replace(".qkv.weight", ".v_proj.weight")
            
            dim = param.shape[0] // 3
            new_state_dict[q_name] = param[:dim]
            new_state_dict[k_name] = param[dim:2*dim]
            new_state_dict[v_name] = param[2*dim:]
            
            logger.debug(f"Split {name} into Q/K/V projections")
            
        elif ".qkv.bias" in name:
            # Split bias similarly
            q_name = name.replace(".qkv.bias", ".q_proj.bias")
            k_name = name.replace(".qkv.bias", ".k_proj.bias")
            v_name = name.replace(".qkv.bias", ".v_proj.bias")
            
            dim = param.shape[0] // 3
            new_state_dict[q_name] = param[:dim]
            new_state_dict[k_name] = param[dim:2*dim]
            new_state_dict[v_name] = param[2*dim:]
            
            logger.debug(f"Split {name} into Q/K/V biases")
        else:
            # Keep other parameters as-is
            new_state_dict[name] = param
    
    # Load the converted state dict
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    
    if missing:
        logger.warning(f"Missing keys when loading weights: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys when loading weights: {unexpected}")
    
    logger.info("Successfully loaded weights with QKV conversion")


# Convenience function for printing LoRA parameters
def print_lora_parameters(model: PeftModel) -> None:
    """
    Print information about LoRA parameters in the model.
    
    Args:
        model: PeftModel with LoRA adapters.
    """
    logger.info("=" * 80)
    logger.info("LoRA Parameter Information (DINO ViT only)")
    logger.info("=" * 80)
    
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            if "lora" in name.lower():
                logger.info(f"  {name}: {param.shape} ({param.numel():,} params)")
    
    logger.info("=" * 80)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    logger.info("=" * 80)


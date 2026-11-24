# EventWorld inference and testing

[中文版](REPRODUCING.zh-CN.md)

This guide reproduces the released **P40530** depth forecasts. The forecasting model reads three contiguous 50 ms event packets. Each horizon uses one forward pass and the same observed history. The test data includes the event images, ordered TESPEC features, and depth/semantic labels, so you can run evaluation immediately after extraction.

The archive and weights have been checked against the reference assets. The CPU scorer reproduces the archived per-case and aggregate metrics exactly. A fresh end-to-end GPU run of the public commands is pending.

## 1. Set up the environment

Use Linux with Python 3.10, Git, `curl`, `tar`, and `sha256sum`. For inference and testing, use one NVIDIA Ampere or later GPU with native BF16 support and a driver compatible with the CUDA 11.8 PyTorch wheels. The reference runs used an A100 40 GB. Install the pinned environment from the repository root.

```bash
git clone https://github.com/yan391/EventWorld.git
cd EventWorld

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt

python -m pip check
python -m eventworld --help
```

Check the GPU selected for inference. The command prints the model, compute capability, and CUDA runtime and checks that BF16 is available natively.

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import torch

assert torch.cuda.is_available(), "PyTorch cannot see a CUDA GPU"
capability = torch.cuda.get_device_capability()
print(torch.cuda.get_device_name(), capability, torch.__version__, torch.version.cuda)
assert capability[0] >= 8 and torch.cuda.is_bf16_supported(), "Native BF16 is required"
PY
```

Set `CUDA_VISIBLE_DEVICES` to the index of the GPU you want to use. CPU-only machines can use `verify-data` and `score`.

To check the scoring rules on CPU, run `python -m unittest discover -s tests`.

## 2. Download the model and test data

| Asset | Save as | Download |
| :--- | :--- | :--- |
| EventWorld P40530 weights · 1.73 GB | `weights/EventWorld-P40530.safetensors` | [Google Drive](https://drive.google.com/file/d/1__Oy_SLwvmeaPiMeYueL2Ab7mz15f09L/view?usp=sharing) |
| EventScape test data · 141 MB | `data/EventScape-test-data.tar.gz` | [Google Drive](https://drive.google.com/file/d/159yCwVb9cn0hP5X1yp42RiW9bJhKVc7E/view?usp=sharing) |
| EventVGGT perception weights | `weights/checkpoint.pt` | [Official EventVGGT release](https://huggingface.co/yinruiRen/EventVGGT/blob/main/checkpoint.pt) |
| EventVGGT pose weights (optional) | `weights/checkpoint_pose.pt` | [Official EventVGGT pose release](https://huggingface.co/yinruiRen/EventVGGT/blob/main/checkpoint_pose.pt) |

P40530 contains the released predictor and fusion parameters. The CLI loads it together with the EventVGGT perception checkpoint, which provides the frozen geometry encoder, decoder, and depth head. Run `mkdir -p weights data outputs` in the repository root. Use the Download button on each Google Drive page, then place the files at the paths in the table. Download `checkpoint.pt` from the official release with the command below, then verify the three required files before extracting.

```bash
curl --fail --location --retry 3 \
  https://huggingface.co/yinruiRen/EventVGGT/resolve/main/checkpoint.pt \
  --output weights/checkpoint.pt.part &&
  mv weights/checkpoint.pt.part weights/checkpoint.pt &&
  (grep -v ' weights/checkpoint_pose\.pt$' docs/SHA256SUMS | sha256sum --check -) &&
  tar -xmzf data/EventScape-test-data.tar.gz -C data &&
  python -m eventworld verify-data --data data/EventScape-test-data
```

The integrity check should report `OK` for all three files listed in [SHA256SUMS](SHA256SUMS) for the depth test. The extracted folder contains `inputs/index.json`, `inputs/events/`, `inputs/features/`, `targets/index.json`, and `assets.json`. The indices describe sample provenance, case identifiers, recording groups, and query timestamps.

## 3. Run a forecast

Run the first released window at four horizons. To select a different window, pass a case ID from `data/EventScape-test-data/inputs/index.json` to `--case`.

```bash
CUDA_VISIBLE_DEVICES=0 python -m eventworld infer \
  --data data/EventScape-test-data \
  --weights weights/EventWorld-P40530.safetensors \
  --eventvggt weights/checkpoint.pt \
  --horizons 200 400 600 800 \
  --output outputs/first-window
```

The output `<case_id>.npz` contains:

| Array | Shape | Meaning |
| :--- | :--- | :--- |
| `horizons_ms` | `[Q]` | Requested times after the last event packet ends |
| `depth` | `[Q, 252, 504]` | Forecast relative depth, one map per horizon |
| `current_depth` | `[252, 504]` | Depth decoded from observed history |

Read the result with NumPy.

```python
from pathlib import Path
import numpy as np

file = next(Path("outputs/first-window").glob("*.npz"))
with np.load(file, allow_pickle=False) as output:
    print(output["horizons_ms"])
    print(output["depth"].shape)
```

To export camera-frame point clouds, add `--intrinsics 256 256 252 126` for the released EventScape crop. It adds `xyz_camera[Q,252,504,3]` in the model's relative depth scale. For different cropped camera frames, supply their actual `fx fy cx cy` values. For camera poses, download the optional `weights/checkpoint_pose.pt`, verify it with `grep ' weights/checkpoint_pose\.pt$' docs/SHA256SUMS | sha256sum --check -`, and add `--camera-checkpoint weights/checkpoint_pose.pt` to `infer`. The resulting `world_to_camera[Q,2,4,4]` stores the last observed and forecast extrinsics for each query's decoder context. Pose translations are scale-ambiguous. For each query, the transform mapping the forecast camera frame into the current camera frame is `E_current @ np.linalg.inv(E_future)`.

### Run inference on your own prepared history

```bash
CUDA_VISIBLE_DEVICES=0 python -m eventworld infer \
  --images prepared/event_0.png prepared/event_1.png prepared/event_2.png \
  --features prepared/tespec_ordered.npz \
  --weights weights/EventWorld-P40530.safetensors \
  --eventvggt weights/checkpoint.pt \
  --horizons 200 600 \
  --output outputs/custom-window
```

Pass three 50 ms packet images from oldest to newest, ending at −100, −50, and 0 ms relative to the query cutoff. Match the released three-channel event-image encoding and use the same 252 × 504 crop. The feature file must contain a finite `ordered` array with shape `[1,3,648,96]`, one row-major 18 × 36 spatial grid for each packet. The features come from the frozen first-stage [TESPEC](https://mhdmohammadi.github.io/TESPEC_webpage/) temporal encoder and must align with those same three packets. This entry point consumes prepared inputs. The EventScape archive supplies paired event images and features ready to use.

## 4. Run the EventScape test

Run a one-window check first. It loads both checkpoints, forecasts four horizons, and scores the predictions against the included labels.

```bash
CUDA_VISIBLE_DEVICES=0 python -m eventworld test \
  --data data/EventScape-test-data \
  --weights weights/EventWorld-P40530.safetensors \
  --eventvggt weights/checkpoint.pt \
  --horizons 200 400 600 800 \
  --limit 1 \
  --output outputs/eventscape-smoke
```

Run the full test with a fresh output path.

```bash
CUDA_VISIBLE_DEVICES=0 python -m eventworld test \
  --data data/EventScape-test-data \
  --weights weights/EventWorld-P40530.safetensors \
  --eventvggt weights/checkpoint.pt \
  --horizons 200 400 600 800 \
  --output outputs/eventscape
```

The test first writes `predictions/*.npz` and a checksum manifest. It then loads the exact-timestamp depth and semantics and writes `scores/per_case.csv` and `scores/summary.json`. A valid pixel has finite positive ground-truth depth no farther than 30 m and a non-sky label. The scorer fits the lower median scale separately at each target and averages metrics within recordings, then averages the recording means. `delta1` is a fraction; multiply by 100 for percent. Use the default four horizons to compare with the README's depth row. The bundle also supplies targets at 120, 160, …, 800 ms for curve reproduction.

| Horizon | Expected AbsRel ↓ | Expected δ₁ ↑ |
| ---: | ---: | ---: |
| 200 ms | 0.0789 | 93.49% |
| 400 ms | 0.0854 | 91.74% |
| 600 ms | 0.1003 | 89.20% |
| 800 ms | 0.1139 | 86.09% |

The case index records the CUDA sampling seed for each window. These are the archived A100 reference values. Minor differences across GPU architectures, CUDA versions, and PyTorch kernels may occur. The score can be recalculated on CPU from a completed prediction bundle.

```bash
python -m eventworld score \
  --data data/EventScape-test-data \
  --predictions outputs/eventscape/predictions \
  --output outputs/eventscape-rescore
```

Choose fresh output directory names if rerunning commands. A partial generation without `generation.json` is not a valid scoring input.

## Other datasets

| Dataset | Official source |
| :--- | :--- |
| EventScape original archives | [University of Zurich Robotics and Perception Group](https://rpg.ifi.uzh.ch/RAMNet.html) |
| DSEC | [Official downloads](https://dsec.ifi.uzh.ch/dsec-datasets/download/) |
| PEOD | [Official dataset page](https://echosliu.github.io/PEOD-dataset/) |
| M3ED | [Official sequences](https://m3ed.io/sequences/) |

The EventScape scorer consumes the release bundle layout. The other datasets retain their upstream formats; the custom inference command expects their inputs to be prepared in the format described above.

## Source and licenses

The MIT license covers original EventWorld code and tests. See the notices for vendored source, checkpoint provenance, and the EventScape and TESPEC attributions. [MIT License](../LICENSE) · [Source and license notices](../THIRD_PARTY_NOTICES.md)

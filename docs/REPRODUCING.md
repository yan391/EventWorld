# EventWorld inference and testing

[中文版](REPRODUCING.zh-CN.md)

This guide reproduces the released **P40530** depth forecasts. The forecasting model reads three contiguous 50 ms event packets. Each horizon uses one forward pass and the same observed history. The test data includes the event images, ordered TESPEC features, and depth/semantic labels, so you can run evaluation immediately after extraction.

The archive and weights have been checked against the reference assets. The CPU scorer reproduces the archived per-case and aggregate metrics exactly. A fresh end-to-end GPU run of the public commands is pending.

## 1. Set up the environment

Use Linux with Python 3.10 and an NVIDIA CUDA GPU that supports BF16. The reference environment uses an A100 40 GB, PyTorch 2.3.1, and CUDA 11.8. Use a writable output directory with enough space for forecasts. Driver compatibility depends on the PyTorch wheel and the GPU.

```bash
git clone https://github.com/yan391/EventWorld.git
cd EventWorld

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt

python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.is_bf16_supported())'
python -m eventworld --help
```

Both CUDA checks should print `True`. Set `CUDA_VISIBLE_DEVICES=0` before launch to select one GPU. CPU-only machines can use `verify-data` and `score`.

To check the scoring rules on CPU, run `python -m unittest discover -s tests`.

## 2. Download the model and test data

| Asset | File | Source |
| :--- | :--- | :--- |
| EventWorld checkpoint | `EventWorld-P40530.safetensors` | Google Drive link pending |
| EventScape test data | `EventScape-test-data.tar.gz` | Google Drive link pending |
| EventVGGT perception checkpoint | `checkpoint.pt` | [Official EventVGGT weights](https://huggingface.co/yinruiRen/EventVGGT/blob/main/checkpoint.pt) |
| EventVGGT pose checkpoint, optional | `checkpoint_pose.pt` | [Official EventVGGT pose weights](https://huggingface.co/yinruiRen/EventVGGT/blob/main/checkpoint_pose.pt) |

Put the weights under `weights/` and extract the archive under `data/`. The paths in the following commands are relative to the Git repository root. Full SHA-256 values are in [SHA256SUMS](SHA256SUMS). The P40530 checkpoint contains the released predictor and fusion parameters. The EventVGGT perception checkpoint provides the frozen geometry encoder, decoder, and depth head.

```bash
mkdir -p weights data outputs
# Place the downloaded EventWorld checkpoint and EventScape archive in weights/ and data/.
curl --fail --location --retry 3 \
  https://huggingface.co/yinruiRen/EventVGGT/resolve/main/checkpoint.pt \
  --output weights/checkpoint.pt
sha256sum --check docs/SHA256SUMS --ignore-missing
tar -xmzf data/EventScape-test-data.tar.gz -C data
python -m eventworld verify-data --data data/EventScape-test-data
```

The extracted folder contains `inputs/index.json`, `inputs/events/`, `inputs/features/`, `targets/index.json`, and `assets.json`. The indices describe sample provenance, case identifiers, recording groups, and query timestamps. The test archive is the processed EventScape subset used by this release.

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

To export camera-frame point clouds, add `--intrinsics 256 256 252 126` for the released EventScape crop. It adds `xyz_camera[Q,252,504,3]` in the model's relative depth scale. For different cropped camera frames, supply their actual `fx fy cx cy` values. To read camera poses, add `--camera-checkpoint weights/checkpoint_pose.pt`. The resulting `world_to_camera[Q,2,4,4]` stores the last observed and forecast extrinsics for each query's decoder context. Pose translations are scale-ambiguous. For each query, the transform mapping the forecast camera frame into the current camera frame is `E_current @ np.linalg.inv(E_future)`.

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

For a quick end-to-end check, append `--limit 1` and choose a new output path. Run the full command to score the entire release bundle.

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

EventWorld uses vendored model components from [EventVGGT](https://github.com/yinruiRen/EventVGGT) and [VGGT-World](https://github.com/SimonSun0810/VGGT-World) matched to the recorded numerical path. Upstream license copies are included for [EventVGGT](../eventworld/_vendor/LICENSES/EventVGGT-MIT.txt) and [VGGT-derived code](../eventworld/_vendor/LICENSES/VGGT-License.txt). EventScape data comes from the [original EventScape project](https://rpg.ifi.uzh.ch/RAMNet.html). Please cite the dataset authors when using these assets.

# EventWorld 推理与测试指南

[English](REPRODUCING.md)

使用发布的 P40530 权重可以在 CUDA GPU 上查询指定时距的未来深度。EventScape 测试数据已包含过去三个 50 毫秒事件包的图像、对应的 TESPEC 特征和评分标签。配置环境、加载权重后，可以直接使用解压后的数据运行推理和测试。

测试数据和权重已通过校验，CPU 评分器已逐样本复现归档指标。公开命令的端到端 GPU 实跑待验收。

## 1. 配置环境

使用 Linux 和 Python 3.10，并准备 Git、`curl`、`tar` 和 `sha256sum`。推理和测试使用一张具备原生 BF16 支持的 NVIDIA Ampere 或更新架构的 GPU，驱动需兼容 CUDA 11.8 的 PyTorch 安装包。归档结果使用 A100 40 GB。下面的命令固定了 PyTorch 和相关依赖的版本。

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

确认 PyTorch 能看到 GPU，并检查原生 BF16 支持。命令会输出显卡型号、计算能力、PyTorch 版本和 CUDA 运行时版本。

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

如果在多卡机器上运行，将 `CUDA_VISIBLE_DEVICES` 换成要用的显卡编号。`verify-data` 和 `score` 可以在 CPU 上运行。可用 `python -m unittest discover -s tests` 检查评分逻辑。

## 2. 下载权重和测试数据

| 资产 | 保存路径 | 下载 |
| :--- | :--- | :--- |
| EventWorld P40530 权重（1.73 GB） | `weights/EventWorld-P40530.safetensors` | [Google Drive](https://drive.google.com/file/d/1__Oy_SLwvmeaPiMeYueL2Ab7mz15f09L/view?usp=sharing) |
| EventScape 测试数据（141 MB） | `data/EventScape-test-data.tar.gz` | [Google Drive](https://drive.google.com/file/d/159yCwVb9cn0hP5X1yp42RiW9bJhKVc7E/view?usp=sharing) |
| EventVGGT 感知权重 | `weights/checkpoint.pt` | [官方权重](https://huggingface.co/yinruiRen/EventVGGT/blob/main/checkpoint.pt) |
| EventVGGT 位姿权重（可选） | `weights/checkpoint_pose.pt` | [官方位姿权重](https://huggingface.co/yinruiRen/EventVGGT/blob/main/checkpoint_pose.pt) |

在仓库根目录运行 `mkdir -p weights data outputs`。从 Google Drive 下载 P40530 和测试数据，并按上表保存。P40530 提供预测器和融合模块，EventVGGT 感知权重提供冻结的几何编码器、解码器和深度头。推理与测试命令会同时加载这两份权重。

```bash
curl --fail --location --retry 3 \
  https://huggingface.co/yinruiRen/EventVGGT/resolve/main/checkpoint.pt \
  --output weights/checkpoint.pt.part &&
  mv weights/checkpoint.pt.part weights/checkpoint.pt &&
  (grep -v ' weights/checkpoint_pose\.pt$' docs/SHA256SUMS | sha256sum --check -) &&
  tar -xmzf data/EventScape-test-data.tar.gz -C data &&
  python -m eventworld verify-data --data data/EventScape-test-data
```

校验时三份必需文件都应显示 `OK`，完整哈希见 [SHA256SUMS](SHA256SUMS)。解压后，`inputs/index.json` 记录输入样本和事件时间戳，`targets/index.json` 记录各预测时距对应的评分目标，`assets.json` 供数据完整性检查使用。

## 3. 推理一个窗口

```bash
CUDA_VISIBLE_DEVICES=0 python -m eventworld infer \
  --data data/EventScape-test-data \
  --weights weights/EventWorld-P40530.safetensors \
  --eventvggt weights/checkpoint.pt \
  --horizons 200 400 600 800 \
  --output outputs/first-window
```

默认选择 `inputs/index.json` 中的第一个窗口。用 `--case <case_id>` 可以指定其他样本。输出为 `outputs/first-window/<case_id>.npz`，用 NumPy 的 `np.load(..., allow_pickle=False)` 读取。

| 数组 | 形状 | 含义 |
| :--- | :--- | :--- |
| `horizons_ms` | `[Q]` | 相对于最后一个事件包结束时刻的预测时距 |
| `depth` | `[Q, 252, 504]` | 每个预测时距的相对深度图 |
| `current_depth` | `[252, 504]` | 从观测历史读出的当前相对深度图 |

发布数据的裁剪坐标可用 `--intrinsics 256 256 252 126` 将预测深度反投影为相机坐标系下的点云。需要位姿时，下载可选的 `weights/checkpoint_pose.pt`，用 `grep ' weights/checkpoint_pose\.pt$' docs/SHA256SUMS | sha256sum --check -` 校验，再给 `infer` 加上 `--camera-checkpoint weights/checkpoint_pose.pt`。输出 `world_to_camera` 保存每次查询的当前帧和预测帧外参。

自有数据可用 `--images` 加三张按时间排列的事件图，以及 `--features` 加对应 TESPEC 特征的 NPZ。图像编码、裁剪和特征格式见[英文指南](REPRODUCING.md#run-inference-on-your-own-prepared-history)。

## 4. 运行 EventScape 测试

先运行一个窗口，确认权重加载、深度预测和评分可以走完整流程。

```bash
CUDA_VISIBLE_DEVICES=0 python -m eventworld test \
  --data data/EventScape-test-data \
  --weights weights/EventWorld-P40530.safetensors \
  --eventvggt weights/checkpoint.pt \
  --horizons 200 400 600 800 \
  --limit 1 \
  --output outputs/eventscape-smoke
```

使用新输出目录运行完整测试。

```bash
CUDA_VISIBLE_DEVICES=0 python -m eventworld test \
  --data data/EventScape-test-data \
  --weights weights/EventWorld-P40530.safetensors \
  --eventvggt weights/checkpoint.pt \
  --horizons 200 400 600 800 \
  --output outputs/eventscape
```

程序先保存 `predictions/*.npz` 和 `predictions/generation.json`，再读取时间戳对齐的真值。逐样本结果写到 `scores/per_case.csv`，汇总写到 `scores/summary.json`。每个预测在 30 米内的有效非天空深度像素上做中位数尺度对齐，录制内取平均后再对录制等权平均。`delta1` 为比例，乘以 100 即为百分数。

| 预测时距 | 归档 AbsRel ↓ | 归档 δ₁ ↑ |
| ---: | ---: | ---: |
| 200 ms | 0.0789 | 93.49% |
| 400 ms | 0.0854 | 91.74% |
| 600 ms | 0.1003 | 89.20% |
| 800 ms | 0.1139 | 86.09% |

不同 GPU 和 CUDA 内核可能带来轻微数值差异。已有完整预测时，可以直接在 CPU 上复算评分。

```bash
python -m eventworld score \
  --data data/EventScape-test-data \
  --predictions outputs/eventscape/predictions \
  --output outputs/eventscape-rescore
```

重跑时为 `--output` 指定新目录。CPU 复算需要预测目录中的 `generation.json` 和相应预测文件。

## 数据集官网

| 数据集 | 官网 |
| :--- | :--- |
| EventScape 原始数据 | [苏黎世大学 RPG 项目页](https://rpg.ifi.uzh.ch/RAMNet.html) |
| DSEC | [官方下载页](https://dsec.ifi.uzh.ch/dsec-datasets/download/) |
| PEOD | [官方数据集页](https://echosliu.github.io/PEOD-dataset/) |
| M3ED | [官方序列目录](https://m3ed.io/sequences/) |

`test` 读取上方发布包的目录结构。其他数据集按各自官网获取，准备好与事件图对应的 TESPEC 特征后，可以通过 `infer --images ... --features ...` 运行自有输入。

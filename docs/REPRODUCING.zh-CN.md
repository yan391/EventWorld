# EventWorld 推理与测试指南

[English](REPRODUCING.md)

使用发布的 P40530 权重可以在 CUDA GPU 上查询指定时距的未来深度。EventScape 测试数据已处理好，包含过去三个 50 毫秒事件包的图像、对应的 TESPEC 特征和评分标签。解压后即可运行。

测试数据和权重已通过校验，CPU 评分器已逐样本复现归档指标。公开命令的端到端 GPU 实跑待验收。

## 安装

参考环境为 Linux、Python 3.10、PyTorch 2.3.1 和 CUDA 11.8。推理使用具备原生 BF16 支持的 NVIDIA Ampere 或更新架构的 GPU。A100 40 GB 是归档结果使用的显卡。

```bash
git clone https://github.com/yan391/EventWorld.git
cd EventWorld
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -m eventworld --help
```

## 下载和解压

| 资产 | 保存路径 | 下载 |
| :--- | :--- | :--- |
| EventWorld 推理权重 | `weights/EventWorld-P40530.safetensors` | Google Drive 链接待补 |
| EventScape 测试数据 | `data/EventScape-test-data.tar.gz` | Google Drive 链接待补 |
| EventVGGT 感知权重 | `weights/checkpoint.pt` | [官方权重](https://huggingface.co/yinruiRen/EventVGGT/blob/main/checkpoint.pt) |
| EventVGGT 位姿权重（可选） | `weights/checkpoint_pose.pt` | [官方位姿权重](https://huggingface.co/yinruiRen/EventVGGT/blob/main/checkpoint_pose.pt) |

```bash
mkdir -p weights data outputs
# 将下载的文件放到上表对应路径。
sha256sum --check docs/SHA256SUMS --ignore-missing
tar -xmzf data/EventScape-test-data.tar.gz -C data
python -m eventworld verify-data --data data/EventScape-test-data
```

## 推理一个窗口

```bash
CUDA_VISIBLE_DEVICES=0 python -m eventworld infer \
  --data data/EventScape-test-data \
  --weights weights/EventWorld-P40530.safetensors \
  --eventvggt weights/checkpoint.pt \
  --horizons 200 400 600 800 \
  --output outputs/first-window
```

默认选择输入索引中的第一个窗口，`--case` 可指定其他样本。输出 `<case_id>.npz` 中，`horizons_ms` 保存各时距，`depth` 是 `[Q,252,504]` 的未来相对深度，`current_depth` 是由观测历史读出的当前深度。在本数据的裁剪坐标下，可加 `--intrinsics 256 256 252 126` 从预测深度反投影点云。加 `--camera-checkpoint weights/checkpoint_pose.pt` 可保存当前帧和未来帧的世界到相机外参。

自有数据可用 `--images` 加三张按时间排列的事件图，以及 `--features` 加对应 TESPEC 特征的 NPZ。图像编码、裁剪和特征格式见[英文指南](REPRODUCING.md#run-inference-on-your-own-prepared-history)。

## 运行完整测试

```bash
CUDA_VISIBLE_DEVICES=0 python -m eventworld test \
  --data data/EventScape-test-data \
  --weights weights/EventWorld-P40530.safetensors \
  --eventvggt weights/checkpoint.pt \
  --horizons 200 400 600 800 \
  --output outputs/eventscape
```

快速验证可以加 `--limit 1` 并换一个输出目录。程序先封存全部预测，然后读取时间戳对齐的真值。结果写到 `scores/per_case.csv` 和 `scores/summary.json`。按 30 米内的非天空像素对每个预测做中位数尺度对齐，在每条录制内汇总后对录制等权平均。结果中的 `delta1` 为比例，乘以 100 得到百分数。完整测试的预期结果、CPU 复算和输出字段见[英文指南](REPRODUCING.md#4-run-the-eventscape-test)。

## 数据集官网

- [EventScape](https://rpg.ifi.uzh.ch/RAMNet.html)
- [DSEC](https://dsec.ifi.uzh.ch/dsec-datasets/download/)
- [PEOD](https://echosliu.github.io/PEOD-dataset/)
- [M3ED](https://m3ed.io/sequences/)

<h1 align="center">EventWorld</h1>
<p align="center"><strong>仅用相机事件预测未来场景几何</strong></p>
<p align="center">
  <strong>Junhao Yan<sup>1</sup>, Dachun Kai<sup>2</sup>, Xiaoyan Sun<sup>2</sup></strong><br>
  <sup>1</sup> 武汉理工大学<br>
  <sup>2</sup> 中国科学技术大学
</p>
<p align="center">
  <a href="#驾驶演示">驾驶演示</a> &nbsp;·&nbsp;
  <a href="#方法">方法</a> &nbsp;·&nbsp;
  <a href="#实验结果">实验结果</a> &nbsp;·&nbsp;
  <a href="#运行-eventworld">推理与测试</a> &nbsp;·&nbsp;
  <a href="#项目视频">项目短片</a> &nbsp;·&nbsp;
  <a href="README.md">English</a>
</p>

![左侧是目标时刻的 RGB 参考画面，右侧是提前 200 毫秒给出的深度预测。下方时间线标出三个历史事件包、观测截止和预测时刻。](media/forecast-preview.jpg)

EventWorld 读取过去 **150 毫秒**的相机事件，用一次前向传播预测指定时刻的**深度、点云和相机位姿**。

| EventScape Town05 | DSEC 低照驾驶 | NVIDIA A100 推理 |
| :--- | :--- | :--- |
| 平均 AbsRel 比 DriveLaW（RGB）低 **19.1%** | 在一条低照录制上，平均 AbsRel 比 VGGT-World（RGB）低 **50.4%** | 200–800 毫秒时距的单次查询约 **0.28 秒** |

## 驾驶演示

左侧是目标时刻的 RGB 参考画面，右侧是提前 **200 毫秒**做出的深度预测。EventWorld 的输入只有观测截止前的事件。

![桥上弯道的 RGB 参考画面和 EventWorld 深度预测。](media/bridge-forecast.gif)

<details>
<summary>城市街道序列</summary>

![城市街道的 RGB 参考画面和 EventWorld 深度预测。](media/urban-forecast.gif)

</details>

## 方法

![EventWorld 方法架构。](media/method.webp)

1. **编码历史。** EventVGGT 和 TESPEC 分别从三个 50 毫秒事件包中提取场景结构和运动特征。
2. **融合与预测。** 每个图像块的几何状态用于给运动特征分配权重。预测器根据融合后的历史状态和指定时距，直接预测未来状态。
3. **解码几何。** 共享解码器结合历史状态和预测状态，恢复未来深度、点云和相机位姿。

训练时，模型用目标时刻的事件构造监督信号，预测模块无需深度或位姿标注。

## 运行 EventWorld

使用 Linux、Python 3.10、PyTorch 2.3.1 的 CUDA 11.8 安装包，以及一张支持原生 BF16 的 NVIDIA GPU。配置环境后，下载两份必需权重和处理好的测试数据。用 `python -m eventworld infer` 推理单个窗口，或用 `python -m eventworld test` 生成预测并计算深度指标。指南提供可直接复制的命令、文件路径、校验步骤和参考结果。[中文推理与测试指南](docs/REPRODUCING.zh-CN.md)

- 模型权重。[EventWorld P40530 权重](https://drive.google.com/file/d/1__Oy_SLwvmeaPiMeYueL2Ab7mz15f09L/view?usp=sharing) · [必需的 EventVGGT 感知权重](https://huggingface.co/yinruiRen/EventVGGT/blob/main/checkpoint.pt) · [可选的 EventVGGT 位姿权重](https://huggingface.co/yinruiRen/EventVGGT/blob/main/checkpoint_pose.pt)
- 处理好的输入和评分标签。[EventScape 测试数据](https://drive.google.com/file/d/159yCwVb9cn0hP5X1yp42RiW9bJhKVc7E/view?usp=sharing)
- 原始数据集官网。[EventScape](https://rpg.ifi.uzh.ch/RAMNet.html) · [DSEC](https://dsec.ifi.uzh.ch/dsec-datasets/download/) · [PEOD](https://echosliu.github.io/PEOD-dataset/) · [M3ED](https://m3ed.io/sequences/)

## 实验结果

- **EventScape Town05 深度。** 在本发布使用的评测子集上，200、400、600、800 毫秒四个时距的平均 AbsRel 为 **0.0946**，相对 RGB 基线 DriveLaW 的 **0.1170** 降低 **19.1%**。平均 δ₁ 为 **90.13%**，DriveLaW 为 **90.17%**。评测对每个预测做中位数尺度对齐，使用 30 米内的非天空像素。[结果 CSV](results/eventscape-depth.csv)
- **EventScape 点云与 M3ED 位姿。** EventScape 在 200 和 800 毫秒的平均点云 F-score 为 **44.65%**，DriveLaW 为 **40.55%**，阈值为参考点云半径的 1%。M3ED 零样本评测中，经逐窗口相似变换对齐后的绝对轨迹误差为 **0.0973 米**，DriveLaW 为 **0.1039 米**。
- **真实低照与高速序列。** 直接将 EventScape 训练的模型用于一个 DSEC 低照录制和一个 PEOD 高速录制。在 100–800 毫秒的八个时距上，相比 VGGT-World（RGB）的平均 AbsRel 分别降低 **50.4%** 和 **59.8%**。DSEC 的参考深度由标定视差得到，PEOD 使用 EventVGGT 生成的伪深度。[结果 CSV](results/zero-shot-depth.csv)

![EventScape 深度预测与有符号相对误差。](media/eventscape-depth.webp)

<details>
<summary>点云和相机位姿预测</summary>

![EventScape 点云预测与参考点云。](media/eventscape-cloud.webp)

![M3ED 预测与实测相机轨迹。](media/m3ed-poses.webp)

</details>

![DSEC 和 PEOD 在八个预测时距上的 AbsRel 曲线，EventWorld 在两条录制的各时距均取得最低误差。](media/transfer-curves.png)

![DSEC 低照和 PEOD 高速场景中的 400 毫秒深度预测误差。](media/transfer-qualitative.webp)

## 项目视频

<img src="media/film-poster.webp" width="640" alt="EventWorld 项目短片封面">

[完整短片（MP4）](media/eventworld-film.mp4) · [英文字幕（VTT）](media/film.en.vtt) · [可逐帧拖动的交互对照](http://40.81.25.249/#driving)

## 致谢

感谢 VGGT-World 和 EventVGGT 的作者公开预训练模型，EventWorld 在这些模型的基础上开展研究。

- **VGGT-World。** 我们用其 KITTI 权重初始化 EventWorld 的预测器，也在实验中将 VGGT-World 作为预测基线。[论文](https://arxiv.org/abs/2603.12655) · [代码](https://github.com/SimonSun0810/VGGT-World)
- **EventVGGT。** 我们用其公开感知权重构建冻结的几何编码器、解码器和深度预测头，相机位姿预测使用其官方位姿权重中的相机预测头。[论文](https://arxiv.org/abs/2603.09385) · [代码](https://github.com/yinruiRen/EventVGGT)

## 许可证

EventWorld 自研代码和测试采用 MIT 许可证。上游代码、模型权重与数据的来源和适用条款见许可说明。[MIT 许可证](LICENSE) · [来源与许可说明](THIRD_PARTY_NOTICES.md)

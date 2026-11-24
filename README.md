# EventWorld

**Forecasting Future Geometry from Events Alone**

EventWorld forecasts what a scene will look like geometrically at a chosen future time. It takes **150 ms of camera events** and produces **depth, point clouds, and camera poses** through a shared geometry decoder.

**[Driving previews](#driving-previews)** · **[Method](#method)** · **[Experimental results](#results)** · **[Project film](#project-film)** · **[中文介绍](README.zh-CN.md)**

![An EventScape street scene at the target time beside EventWorld's 200 ms depth forecast.](media/forecast-preview.jpg)

## Driving previews

Each clip shows the target-time RGB reference alongside EventWorld's depth forecast at **+200 ms**. The forecast uses only events recorded before the observation cutoff.

![Urban street sequence showing synchronized target-time RGB and forecast depth at 200 ms.](media/urban-forecast.gif)

<details>
<summary>Bridge bend sequence</summary>

![Bridge sequence showing synchronized target-time RGB and forecast depth at 200 ms.](media/bridge-forecast.gif)

</details>

## Method

Camera events capture brightness changes. Moving edges reveal scene structure and motion, including in recordings where RGB frames are dark or blurred. EventWorld learns to combine these signals and query the future at a supplied horizon.

![EventWorld architecture showing the two event encoders, geometry-guided fusion, horizon-conditioned predictor, and geometry outputs.](media/method.webp)

1. **Encode the event history.** Frozen EventVGGT and TESPEC encoders extract geometric states and temporal features from three 50 ms event packets.
2. **Fuse and forecast.** Geometry guides the weighting of temporal features at each patch. The predictor takes the fused history and the requested horizon and predicts that future state in a single forward pass.
3. **Decode the geometry.** A shared frozen decoder reads the forecast state alongside the original historical states. The training target comes from encoding events at the target time, so training the forecaster requires no depth or pose annotations.

## Results

### Depth on held-out EventScape Town05

EventWorld has the lowest **AbsRel** at all four reported horizons. Relative to DriveLaW, the strongest RGB baseline by mean AbsRel, it lowers mean AbsRel by **19.1%** and reaches a comparable mean δ₁ (**90.13%** versus **90.17%**).

| Method | Input | 200 ms | 400 ms | 600 ms | 800 ms | Mean AbsRel ↓ | Mean δ₁ ↑ |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **EventWorld** | Events | **0.0789** | **0.0854** | **0.1003** | **0.1139** | **0.0946** | 90.13% |
| DriveLaW | RGB | 0.0796 | 0.1090 | 0.1280 | 0.1515 | 0.1170 | **90.17%** |
| VGGT-World | RGB | 0.0917 | 0.1131 | 0.1862 | 0.1645 | 0.1389 | 87.09% |
| Copy-last | Events | 0.1202 | 0.1616 | 0.1903 | 0.2122 | 0.1711 | 78.30% |

The four horizon columns report AbsRel. The evaluation uses 238 windows from the Town05 test archive and scores non-sky reference depths within 30 m after per-prediction median scale alignment. [Complete EventScape depth table (CSV)](results/eventscape-depth.csv)

![EventScape depth forecasts, ground truth, and signed depth errors in two scenes at 200 and 800 ms.](media/eventscape-depth.webp)

### Point clouds and camera motion

| Task | Protocol | EventWorld | DriveLaW (RGB) |
| :--- | :--- | ---: | ---: |
| Point-cloud forecasting | Mean normalized Chamfer distance on EventScape, 200 and 800 ms ↓ | **0.0532** | 0.0590 |
| Point-cloud forecasting | Mean F-score at 1% of the reference cloud radius, 200 and 800 ms ↑ | **44.65%** | 40.55% |
| Camera-pose forecasting | Absolute trajectory error on M3ED, zero-shot, after per-window similarity alignment ↓ | **0.0973 m** | 0.1039 m |

<details>
<summary>EventScape point-cloud forecasts</summary>

![Reference and predicted point clouds from EventScape at 200, 400, and 800 ms.](media/eventscape-cloud.webp)

</details>

<details>
<summary>M3ED camera-pose forecasts</summary>

![Ground-truth and predicted M3ED camera trajectories through 800 ms for two examples.](media/m3ed-poses.webp)

</details>

### Transfer to real recordings

| Recording | Windows | Depth reference | Mean AbsRel reduction vs. VGGT-World (RGB) |
| :--- | ---: | :--- | ---: |
| DSEC low-light driving | 83 | Calibrated disparity, 30 m and common visibility | **50.4%** |
| PEOD high-speed motion | 73 | Target-time EventVGGT pseudo-depth | **59.8%** |

EventWorld transfers from EventScape without fine-tuning on these recordings. The reductions average eight forecast horizons from 100 to 800 ms. [Download the zero-shot depth table (CSV)](results/zero-shot-depth.csv)

![AbsRel versus forecast horizon on DSEC and PEOD. EventWorld has the lowest reported error at each of the eight horizons on both recordings.](media/transfer-curves.png)

![Examples of DSEC low-light driving and PEOD high-speed motion with signed relative depth errors at a 400 ms forecast horizon.](media/transfer-qualitative.webp)

### Efficiency and scope

On an NVIDIA A100, inference takes about **0.28 s per query** across the measured 200–800 ms horizons. The encoded history can be shared across queries. EventWorld returns a single prediction at each requested horizon. When both the camera and the scene are still, sparse events limit what it can observe.

## Project film

The film covers the method, both driving sequences, and the forecasting experiments.

![EventWorld project film poster.](media/film-poster.webp)

[Full film (.mp4)](media/eventworld-film.mp4) · [English captions (.vtt)](media/film.en.vtt) · [Interactive comparison](http://40.81.25.249/#driving)

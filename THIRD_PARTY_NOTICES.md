# Source and license notices

The [EventWorld MIT license](LICENSE) covers original EventWorld source code and tests. The following notices identify the upstream sources and the scope of the downloadable research assets.

## Vendored source code

| Component | Location | Upstream terms and notices |
| :--- | :--- | :--- |
| EventVGGT event encoders and geometry readouts | [`eventworld/_vendor/vggt/`](eventworld/_vendor/vggt/) | The [EventVGGT repository](https://github.com/yinruiRen/EventVGGT) publishes its contributions under [MIT](eventworld/_vendor/LICENSES/EventVGGT-MIT.txt). Individual source files retain their upstream notices for inherited code. |
| VGGT and VGGT-World implementations | [`eventworld/_vendor/author_fm.py`](eventworld/_vendor/author_fm.py) and corresponding code under [`eventworld/_vendor/vggt/`](eventworld/_vendor/vggt/) | [VGGT-World](https://github.com/SimonSun0810/VGGT-World#license) adopts the [VGGT License](eventworld/_vendor/LICENSES/VGGT-License.txt) unless otherwise indicated. Files marked Apache-2.0 retain their original notices and the [Apache License 2.0 text](eventworld/_vendor/LICENSES/Apache-2.0.txt). |

The vendored snapshot is scoped to the components used for EventWorld inference, with imports and inference paths adapted for this release. Cite the upstream projects when using their components. [VGGT](https://github.com/facebookresearch/vggt) · [VGGT-World](https://github.com/SimonSun0810/VGGT-World) · [EventVGGT](https://github.com/yinruiRen/EventVGGT)

## Checkpoints

`EventWorld-P40530.safetensors` is the checkpoint trained and released by the EventWorld team. It contains the forecasting and fusion parameters, including a forecasting backbone initialized from VGGT-World's KITTI checkpoint. The EventWorld source code's MIT grant applies solely to the original source and tests. Consult the applicable upstream terms for use or redistribution of pretrained or derived weights. [VGGT-World checkpoints and license](https://github.com/SimonSun0810/VGGT-World#checkpoints)

The perception checkpoint `checkpoint.pt` and optional pose checkpoint `checkpoint_pose.pt` come directly from the upstream model release. Their model card identifies the release as MIT. [Official EventVGGT model release](https://huggingface.co/yinruiRen/EventVGGT)

## EventScape test data

The EventWorld team prepared the evaluation archive from EventScape Town05 recordings. It contains event images derived from EventScape, depth and semantic annotations from EventScape, and motion features extracted with the pretrained TESPEC encoder. Cite Gehrig et al. for EventScape and Mohammadi et al. for TESPEC when using the corresponding assets. [EventScape and citation](https://rpg.ifi.uzh.ch/RAMNet.html) · [TESPEC and citation](https://mhdmohammadi.github.io/TESPEC_webpage/)

The original data remains subject to the source provider's terms. The official EventScape page provides download links and a citation, while permission to redistribute the processed archive remains to be confirmed. Consult the dataset provider before mirroring or repackaging it. [EventScape provider](https://rpg.ifi.uzh.ch/RAMNet.html)

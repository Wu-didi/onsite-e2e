
# 安装依赖
1. 首先安装python3.10

2. 代码运行需要安装**nvidia driver**和**cuda11.1**，请自行安装。

3. 安装网络库`pip install dists/libMulticastNetwork-1.0.0-cp310-cp310-linux_x86_64.whl`

4. 安装torch

`pip install torch=1.10.1+cu111`

`pip install torchvision=0.11.2+cu111`


5. 安装mmdetection

`pip install -U openmim`

`mim install mmengine`

`mim install 'mmcv>=2.0.0rc4'`

`mim install 'mmdet>=3.0.0'`

`mim install "mmdet3d>=1.1.0"`

6. 下载demo模型的权重和配置文件

lidar感知

`mim download mmdet3d --config pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car --dest .`

camera感知

`mim download mmdet --config yolox_tiny_8x8_300e_coco --dest .`

# 使用方法
见使用手册说明


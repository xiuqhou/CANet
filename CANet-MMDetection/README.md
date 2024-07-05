## CANet based on MMDetection

Code is tested with `python=3.10`, `pytorch=1.12.1`, `torchvision=0.13.1`, `mmcv-full=1.7.0`, `mmdet=2.25.2`, other version may also work.

- Change Directory to `CANet-MMDetection`:
  ```shell
  cd CANet-MMDetection
  ```
- Install Pytorch:
  ```
  conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
  ```
- Install MMCV and MMDetection:
  ```shell
  pip install -U openmim
  mim install mmcv-full=1.7.0
  pip install -v -e .
  ```
- Prepare datasets and put them in the `data/` directory:
  ```shell
    data/
     ├─coco/
     │  ├── train2017/
     │  ├── val2017/
     │  └── annotations/
     │
     ├─neu_det_coco/
     │  ├── train2017/
     │  ├── val2017/
     │  └── annotations/
     │
     └─VOCdevkit
        ├── VOC2007/
        └── VOC2012/
  ```
- Train:
  ```shell
  # train CANet on VOC dataset
  python tools/train.py configs/canet/canet_r50_fpn_1x_voc0712.py
  # train CANet on COCO dataset
  python tools/train.py configs/canet/canet_r50_fpn_1x_coco.py
  # train CANet on NEU-DET dataset
  python tools/train.py configs/canet/canet_r50_fpn_1x_neu_det_coco.py
  ```
- Test:
  ```shell
  # test CANet on VOC dataset
  python tools/test.py configs/canet/canet_r50_fpn_1x_voc0712.py <path/to/checkpoint.pth> --eval mAP
  # test CANet on COCO dataset
  python tools/test.py configs/canet/canet_r50_fpn_1x_coco.py <path/to/checkpoint.pth> --eval bbox
  # test CANet on NEU-DET dataset
  python tools/test.py configs/canet/canet_r50_fpn_1x_neu_det_coco.py <path/to/checkpoint.pth> --eval bbox
  ```
- See documents under the `docs/` directory for other usage.
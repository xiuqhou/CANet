## CANet based on PyTorch

- Paper: [CANet: Contextual information and spatial attention based network for detecting small defects in manufacturing industry](https://www.sciencedirect.com/science/article/abs/pii/S0031320323002583)
- Author: [Xiuquan Hou](https://github.com/xiuqhou), [Meiqin Liu](https://scholar.google.com/citations?user=T07OWMkAAAAJ&hl=zh-CN&oi=ao), Senlin Zhang, [Ping Wei](https://scholar.google.com/citations?user=1OQBtdcAAAAJ&hl=zh-CN&oi=ao), [Badong Chen](https://scholar.google.com/citations?user=mq6tPX4AAAAJ&hl=zh-CN&oi=ao).

**💖 If our CANet is helpful to your researches or projects, please help star this repository. Thanks! 🤗**


### Installation

- Change Directory to `CANet-MMDetection`:
  ```shell
  cd CANet-pytorch
  ```
- Install Pytorch and other requirements, other version may also work:
  ```
  conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
  pip install -r requirements.txt
  ```


### Prepare Dataset

Download [COCO 2017](https://cocodataset.org/) and put them in the `data/` folder. You can use [`tools/visualize_datasets.py`](tools/visualize_datasets.py) to visualize the dataset annotations.

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
    └─pascal_voc_cocostyle
    ├── VOC2007/
    └── VOC2012/
```

<details>

<summary>Example for visualization</summary>

```shell
python tools/visualize_datasets.py \
    --coco-img data/coco/val2017 \
    --coco-ann data/coco/annotations/instances_val2017.json \
    --show-dir visualize_dataset/
```

</details>

### Train a model

Modify parameters in [`configs/train_config.py`](configs/train_config.py). Use `CUDA_VISIBLE_DEVICES` to specify GPU/GPUs and start training.

```shell
CUDA_VISIBLE_DEVICES=0 accelerate launch main.py    # train with 1 GPU
CUDA_VISIBLE_DEVICES=0,1 accelerate launch main.py  # train with 2 GPUs
```

<details>

<summary>A simple example for train config</summary>

```python
from torch import optim

from datasets.coco import CocoDetection
from transforms import presets
from optimizer import param_dict

# Commonly changed training configurations
num_epochs = 12   # train epochs
batch_size = 2    # total_batch_size = #GPU x batch_size
num_workers = 4   # workers for pytorch DataLoader
pin_memory = True # whether pin_memory for pytorch DataLoader
print_freq = 50   # frequency to print logs
starting_epoch = 0
max_norm = 0.1    # clip gradient norm

output_dir = None  # path to save checkpoints, default for None: checkpoints/{model_name}
find_unused_parameters = False  # useful for debugging distributed training

# define dataset for train
coco_path = "data/coco"                # /PATH/TO/YOUR/COCODIR
train_transform = presets.flip_resize  # see transforms/presets to choose a transform
train_dataset = CocoDetection(
    img_folder=f"{coco_path}/train2017",
    ann_file=f"{coco_path}/annotations/instances_train2017.json",
    transforms=train_transform,
    train=True,
)
test_dataset = CocoDetection(
    img_folder=f"{coco_path}/val2017",
    ann_file=f"{coco_path}/annotations/instances_val2017.json",
    transforms=None,  # the eval_transform is integrated in the model
)

# model config to train
model_path = "configs/canet/canet_resnet50_800_1333.py"

# specify a checkpoint folder to resume, or a pretrained ".pth" to finetune, for example:
# checkpoints/canet_resnet50_800_1333/train/2024-03-22-09_38_50
# checkpoints/canet_resnet50_800_1333/train/2024-03-22-09_38_50/best_ap.pth
resume_from_checkpoint = None

learning_rate = 1e-2  # initial learning rate
optimizer = optim.SGD(lr=learning_rate, weight_decay=1e-4, betas=(0.9, 0.999))
lr_scheduler = optim.lr_scheduler.MultiStepLR(milestones=[10], gamma=0.1)

# This define parameter groups with different learning rate
param_dicts = param_dict.finetune_backbone_and_linear_projection(lr=learning_rate)
```
</details>

### Evaluation/Test

To evaluate a model with one or more GPUs, specify `CUDA_VISIBLE_DEVICES`, `dataset`, `model` and `checkpoint`.

```shell
CUDA_VISIBLE_DEVICES=<gpu_ids> accelerate launch test.py --coco-path /path/to/coco --model-config /path/to/model.py --checkpoint /path/to/checkpoint.pth
```

Optional parameters are as follows, see [test.py](test.py) for full parameters:

- `--show-dir`: path to save detection visualization results.
- `--result`: specify a file to save detection numeric results, end with `.json`.

<details>

<summary>An example for evaluation</summary>

To evaluate `canet_resnet50_800_1333` on `coco` using 8 GPUs, save predictions to `result.json` and visualize results to `visualization/`:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch test.py
    --coco-path data/coco \
    --model-config configs/canet/canet_resnet50_800_1333.py \
    --checkpoint <path/to/checkpoint> \
    --result result.json \
    --show-dir visualization/
```

</details>

<details>

<summary>Evaluate a json result file</summary>

To evaluate the json result file obtained above, specify the `--result` but not specify `--model`.

```shell
CUDA_VISIBLE_DEVICES=0 accelerate launch test.py --coco-path /path/to/coco --result /path/to/result.json
```

Optional parameters, see [test.py](test.py) for full parameters:

- `--show-dir`: path to save detection visualization results.

</details>

### Inference

Use [`inference.py`](inference.py) to perform inference on images. You should specify the image directory using `--image-dir`.

```shell
python inference.py --image-dir /path/to/images --model-config /path/to/model.py --checkpoint /path/to/checkpoint.pth --show-dir /path/to/dir
```

<details>

<summary>An example for inference on an image folder</summary>

To performa inference for images under `images/` and save visualizations to `visualization/`:

```shell
python inference.py \
    --image-dir images/ \
    --model-config configs/canet/canet_resnet50_800_1333.py \
    --checkpoint checkpoint.pth \
    --show-dir visualization/
```

</details>

See [`inference.ipynb`](inference.ipynb) for inference on single image and visualization.

### Benchmark a model

To test the inference speed, memory cost and parameters of a model, use `tools/benchmark_model.py`.

```shell
python tools/benchmark_model.py --model-config configs/canet/canet_resnet50_800_1333.py
```

### Train your own datasets

To train your own datasets, there are some things to do before training:

1. Prepare your datasets with COCO annotation format, and modify `coco_path` in [`configs/train_config.py`](configs/train_config.py) accordingly.
2. Open model configs under [`configs/canet`](configs/canet) and modify the `num_classes` to a number  larger than `max_category_id + 1` of your dataset. For example, from the following annotation in `instances_val2017.json`, we can find the maximum category_id is `90` for COCO, so we set `num_classes = 91`.

    ```json
    {"supercategory": "indoor","id": 90,"name": "toothbrush"}
    ```
    You can simply set `num_classes` to a large enough number if not sure what to set. (For example, `num_classes = 92` or `num_classes = 365` also work for COCO.)
3. If necessary, modify other parameters in model configs under [`configs/canet`](configs/canet/) and [`train_config.py`](train_config.py).

### Export an ONNX model

Change `antialias=False` in line 74 in [base_detector.py](models/detectors/base_detector.py#L74), and run the following shell to export ONNX file. See `ONNXDetector` in [`tools/pytorch2onnx.py`](tools/pytorch2onnx.py) for inference using the ONNX file.

```shell
python tools/pytorch2onnx.py \
    --model-config /path/to/model.py \
    --checkpoint /path/to/checkpoint.pth \
    --save-file /path/to/save.onnx \
    --simplify \  # use onnxsim to simplify the exported onnx file
    --verify  # verify the error between onnx model and pytorch model
```

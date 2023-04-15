# CVPDL_hw1

To start working on this assignment, you should clone this repository into your local machine by using the following command.
    
    git clone git@github.com:benlin1211/CVPDL_hw1.git
# Device

NVIDIA GeForce RTX 3090.
    
# Run train code - DETR

### Create environment
    
    cd detr
    conda create --name detr python=3.10
    conda activate detr
    pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html 
    pip install torchvision==0.12.0
    # pip install -f https://download.pytorch.org/whl/cu110/torch_stable.html torch==1.7.0+cu110 torchvision==0.8.0
    conda install cython scipy
    pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
    pip install -r requirements.txt
    
### Train from pre-train weights
    python main.py --backbone resnet50 --coco_path ../hw1_dataset --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --output_dir ./ckpts_resnet50
    
    python main.py --backbone resnet50 --coco_path ../hw1_dataset --resume https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth --output_dir ./ckpts_resnet50

    python main.py --backbone resnet101 --coco_path ../hw1_dataset --resume https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth --output_dir ./ckpts_resnet101

### Train from your own checkpoints
    python main.py --backbone resnet50 --coco_path ../hw1_dataset --resume ./ckpts_resnet50/checkpoint.pth --output_dir ./ckpts_resnet50
    python main.py --backbone resnet101 --coco_path ../hw1_dataset --resume ./ckpts_resnet101/checkpoint.pth --output_dir ./ckpts_resnet101
    export CUDA_VISIBLE_DEVICES=0,1

### Eval and generate pred_eval.json
    python main.py --batch_size 1 --backbone resnet50 --no_aux_loss --eval --coco_path ../hw1_dataset --resume ./ckpts_resnet50/checkpoint.pth
    python main.py --batch_size 1 --backbone resnet101 --no_aux_loss --eval --coco_path ../hw1_dataset --resume ./ckpts_resnet101/checkpoint.pth
    python ../hw1_dataset/check_your_prediction_valid.py ./pred_eval.json ../hw1_dataset/valid/_annotations.coco.json

### Test and generate pred_eval.json 
    python main.py --batch_size 1 --backbone resnet50 --no_aux_loss --test --coco_path ../hw1_dataset --resume ./ckpts_resnet50/checkpoint.pth
    python main.py --batch_size 1 --backbone resnet101 --no_aux_loss --test --coco_path ../hw1_dataset --resume ./ckpts_resnet101/checkpoint.pth
    
_______________________
# Run train code - yolov8

### Create environment

    cd yolov8
    conda create --name yolov8 python=3.10
    conda activate yolov8
    pip install -r requirements.txt
    
### Train from pre-train weights
    
    python main.py
    
### Train from your own checkpoints
    
    python main.py --resume ./runs/detect/train/weights/best.pt

### Eval and generate pred_eval.json
    # python yolo2submit.py --json_file ./runs/detect/val/predictions.json --out_file ./pred_eval.json
    python main.py --resume ./runs/detect/train/weights/best.pt --eval
    python ../hw1_dataset/check_your_prediction_valid.py ./pred_eval.json ../hw1_dataset/valid/_annotations.coco.json
    
### List all environments

    conda info --envs
    
### Check all package environment

    conda list -n <env_name>

### Close an environment

    conda deactivate

### Remove an environment

    conda env remove -n <env_name>

# Reference:
DETR: https://github.com/facebookresearch/detr
YOLOv8: https://github.com/ultralytics/ultralytics

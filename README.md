# CVPDL_hw1

To start working on this assignment, you should clone this repository into your local machine by using the following command.
    
    git clone git@github.com:benlin1211/CVPDL_hw1.git

    
# Run train code - DETR

### Create environment
You can run the following command to install all the packages listed in the requirements.txt:

    conda create --name detr python=3.8
    conda activate detr
    pip install -f https://download.pytorch.org/whl/cu110/torch_stable.html torch==1.7.0+cu110 torchvision==0.8.0
    conda install cython scipy
    pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
    pip install -r requirements.txt
    
Then, go to "~/anaconda3/envs/detr/lib/python3.8/site-packages/pycocotools/cocoeval.py", line 378, in accumulate, change the line:
    
    # old
    #tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
    #fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
    # new
    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float32)
    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float32)
    
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
    
    
# Run train code - yolov8

    conda create --name yolov8 python=3.8
    conda activate yolov8
    pip install ultralytics

    
### List all environments

    conda info --envs
    
### Check all package environment

    conda list -n detr

### Close an environment

    conda deactivate

### Remove an environment

    conda env remove -n detr

# Reference:
DETR: https://github.com/facebookresearch/detr
YOLOv8: https://github.com/ultralytics/ultralytics
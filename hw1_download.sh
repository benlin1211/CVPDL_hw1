#!/bin/bash

# conda create --name yolov8 python=3.10 -y
# conda activate yolov8
# pip install -r ./yolov8/requirements.txt

# download dataset
# wget https://drive.google.com/file/d/1jjuI7Me9VFhpMHp2QP5gHKU5NPkzCQZk/view?usp=sharing
wget --load-cookies cookies.txt -O hw1_dataset.zip \
     'https://docs.google.com/uc?export=download&id='1jjuI7Me9VFhpMHp2QP5gHKU5NPkzCQZk'&confirm='$(<confirm.txt)

# unzip dataset
unzip -o ./hw1_dataset.zip -d ./
rm -rf ./hw1_dataset.zip

# download checkpoint
wget --load-cookies cookies.txt -O runs.zip \
     'https://docs.google.com/uc?export=download&id='1uvoyhFzVgIRmUmKBAkRK1KZcqWkIcrR1'&confirm='$(<confirm.txt)

# unzip checkpoint to correct directory
unzip -o ./runs.zip -d ./yolov8/
rm -rf ./runs.zip

# bash yolov8/make_yolo_dataset.sh
python ./yolov8/utils/json2yolo.py --coco_path="./hw1_dataset" --save_dir="./yolov8/datasets/hw1_dataset_yolo"
python ./yolov8/utils/yaml_generator.py --coco_path="./hw1_dataset/train/_annotations.coco.json" --data_path="./yolov8/datasets/hw1_dataset_yolo"  --out_path="./yolov8/hw1_dataset_yolo.yaml" 

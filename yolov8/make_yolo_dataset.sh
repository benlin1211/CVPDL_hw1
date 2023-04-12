#!/bin/bash

python utils/json2yolo.py --coco_path="../hw1_dataset" --save_dir="./datasets/hw1_dataset_yolo"
#python utils/json2yolo.py --coco_path="../hw1_dataset" --save_dir_train="./datasets/hw1_dataset_yolo_train" --save_dir_valid="./datasets/hw1_dataset_yolo_valid"

#python utils/yaml_generator.py --coco_path="../hw1_dataset/train/_annotations.coco.json" --data_path="./hw1_dataset_yolo"  --out_path="./hw1_dataset_yolo.yaml"  
python utils/yaml_generator.py --coco_path="../hw1_dataset/train/_annotations.coco.json" --data_path="datasets/hw1_dataset_yolo"  --out_path="./hw1_dataset_yolo.yaml" 
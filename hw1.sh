#!/bin/bash

# ${1}: testing images directory with images named 'xxxx.jpg (e.g. input/test_dir/)
# ${2}: path of output json file (e.g. output/pred.json)
# inference

# bash hw1.sh ./yolov8/datasets/hw1_dataset_yolo/images/test ./output/pred_test.json
# bash hw1.sh ./yolov8/datasets/hw1_dataset_yolo/images/val ./output/pred_eval.json
# python ./hw1_dataset/check_your_prediction_valid.py ./output/pred_eval.json ./hw1_dataset/valid/_annotations.coco.json
python ./yolov8/main.py --test_path $1 --out_path $2 --resume ./yolov8/runs/detect/train/weights/best.pt --test


# python ./main.py --resume ./runs/detect/train/weights/best.pt --test --test_path ./datasets/hw1_dataset_yolo/images/test
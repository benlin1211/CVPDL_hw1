#!/bin/bash

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


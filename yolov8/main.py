import argparse
from ultralytics import YOLO
import numpy as np
import torch
import random

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--seed', default=42, type=int) 
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--test', action='store_true')
    return parser


def main(args):
    # Load a model
    # model = YOLO("yolov8n.yaml")  # build a new model from scratch
    seed = args.seed 
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Use the model
    model.train(data="./hw1_dataset_yolo.yaml", epochs=200)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    print(results)
    success = model.export(format="onnx")  # export the model to ONNX format
    print(success)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

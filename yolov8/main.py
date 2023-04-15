# For more args, please infer to https://docs.ultralytics.com/usage/cfg/#modes
# https://huggingface.co/keremberke/yolov8m-valorant-detection
# https://docs.ultralytics.com/modes/predict/#sources
import argparse
from ultralytics import YOLO
import numpy as np
import torch
import random
import glob
from PIL import Image
import os
import torchvision.transforms as transforms
import json
from tqdm import tqdm

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    # parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--seed', default=42, type=int) 
    # ===================== Train Config ====================
    parser.add_argument('--epochs', default=300, type=int) 
    parser.add_argument('--batch', default=16, type=int)
    parser.add_argument('--lr0', default=1e-2, type=float)
    parser.add_argument('--lrf', default=1e-2, type=float)
    #  https://docs.ultralytics.com/modes/train/#arguments
    # ===================== Eval Config =====================
    parser.add_argument('--resume', default='', help='resume from checkpoint (xxx.pt)') #./runs/detect/train/weights/best.pt
    parser.add_argument('--confidence', default=0.25, type=float) 
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval_path', default='./datasets/hw1_dataset_yolo/images/val', type=str)
    # ===================== Test Config =====================
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_path', default='./datasets/hw1_dataset_yolo/images/test', type=str)
    return parser

def generate_result(model, data_path, confidence):

    results = {}
    pbar = tqdm(glob.glob(os.path.join(data_path, "*.jpg")))
    for img_name in pbar:
        img = Image.open(img_name)
        pred = model(img, conf=args.confidence, verbose=False)[0]
        # print("res:",pred.boxes.xyxy, pred.boxes.cls, pred.boxes.conf)
        
        boxes = pred.boxes.xyxy.cpu().numpy()
        labels = pred.boxes.cls.cpu().numpy()
        scores = pred.boxes.conf.cpu().numpy()

        # do some operation
        # keep_idx = np.where(scores>=confidence)
        # boxes = boxes[keep_idx]
        # labels = labels[keep_idx]
        # scores = scores[keep_idx]    

        res = {} 
        res['boxes'] = boxes.tolist() 
        res['labels'] = labels.tolist() 
        res['scores'] = scores.tolist() 
        results[img_name.split("/")[-1]] = res
    
    return results

def main(args):
    # Fix seed
    seed = args.seed 
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if args.eval:
        print(f"eval, resume from {args.resume}, confidence={args.confidence}")
        model = YOLO(args.resume)
        # https://docs.ultralytics.com/modes/predict/#sources
        # metrics = model.val(batch=16, save_json=True, conf=args.confidence)
        results = generate_result(model, args.eval_path, args.confidence)
        with open('./pred_eval.json', 'w') as fp:
            json.dump(results, fp)
        print("over")
        return

    if args.test:
        print("test, resume from {args.resume}")
        model = YOLO(args.resume)
        results = generate_result(model, args.test_path, args.confidence)
        with open('./pred_test.json', 'w') as fp:
            json.dump(results, fp)
        print("over")
        return
    
    # Load a model
    if args.resume:
        model = YOLO(args.resume)
        # For more args, please infer to https://docs.ultralytics.com/usage/cfg/#modes
    else:  
        model = YOLO("yolov8x.pt")  # load a pretrained model (recommended for training)

    # Train the model
    model.train(data="./hw1_dataset_yolo.yaml", 
                epochs=args.epochs, 
                batch=args.batch, 
                lr0=args.lr0, 
                lrf=args.lrf)  # train the model

    # save the model to ONNX format
    success = model.export(format="onnx")  
    print(success)

    # metrics = model.val()  # evaluate model performance on the validation set
    # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    # print(results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)


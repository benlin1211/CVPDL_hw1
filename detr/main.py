# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset, build_test_set
from engine import evaluate, test, train_one_epoch
from models import build_model
import matplotlib.pyplot as plt
from PIL import Image

import torch.nn.functional as F
from util.box_ops import box_cxcywh_to_xyxy
import matplotlib

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=600, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--num_classes', default=8, type=int,
                        help="Highest category number+1")
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='./ckpts',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    # print(criterion)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    dataset_test = build_test_set(folder_name='test', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_test = DataLoader(dataset_test, args.batch_size, 
                                 drop_last=False, num_workers=args.num_workers) 
    # print(len(data_loader_test)) # 63/2
    # print(len(data_loader_val)) # 128/2

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        # checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        # model_without_ddp.detr.load_state_dict(checkpoint['model'])
        pass

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        # print(checkpoint.keys())
        model_without_ddp_dict = model_without_ddp.state_dict()
        # 1. filter out unnecessary keys
        filtered_dict = {k: v for k, v in checkpoint['model'].items() if k in model_without_ddp_dict}
        # 2. overwrite entries in the existing state dict
        model_without_ddp_dict.update(filtered_dict) 
        # 3. load the new state dict
        model_without_ddp.load_state_dict(model_without_ddp_dict)

        # model_without_ddp.load_state_dict(checkpoint['model'])
        
        if not args.eval and not args.test and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        print("eval")
        assert args.batch_size == 1, "batch_size in eval mode should be 1."
        _, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        my_dataset_val = build_test_set(folder_name='valid', args=args)
        my_data_loader_val = DataLoader(my_dataset_val, args.batch_size, 
                                        drop_last=False, num_workers=args.num_workers) 
    
        results = test(model, my_data_loader_val, device)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
            # TODO: Do the same thing as in test: dump json file.
            # Save results as .json
            with open('pred_eval.json', 'w') as fp:
                json.dump(results, fp)
            print("over")
        return
    
    if args.test:
        print("test")
        assert args.batch_size == 1, "batch_size in test mode should be 1."
        results = test(model, data_loader_test, device)
        if args.output_dir:
            # Save results as .json
            # out_file = os.path.join(args.output_dir, 'pred_test.json') 
            with open('pred_test.json', 'w') as fp:
                json.dump(results, fp)
            print("over")
        return
    
    if args.report: 
        # plt.figure()
        # plt.imshow()
        from torchvision import transforms
        report_img_name = os.path.join(args.coco_path, "test/IMG_2574_jpeg_jpg.rf.ca0c3ad32384309a61e92d9a8bef87b9.jpg")
        img_PIL = Image.open(report_img_name).convert('RGB')
        tf = transforms.Compose([transforms.ToTensor(), 
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        img = tf(img_PIL).to(device)
        img = torch.unsqueeze(img, axis=0)
        h, w = img.shape[-2:]
        # print(img.shape)

        outputs = model(img) 
        outputs = {k: v.detach().cpu() for k, v in outputs.items()}
        pred_logits = F.softmax(outputs['pred_logits'][0], dim=1) # softmax
        pred_boxes = box_cxcywh_to_xyxy(outputs['pred_boxes'][0] * torch.tensor([w, h, w, h], dtype=torch.float32))

        all_boxes = pred_boxes.numpy() #.tolist()
        all_labels = torch.argmax(pred_logits, axis=1).numpy() #.tolist()
        all_scores = torch.amax(pred_logits, axis=1).numpy() #.tolist()
        nonempty_index = np.where(all_labels!=8)
        all_boxes = all_boxes[nonempty_index]
        all_labels = all_labels[nonempty_index]
        all_scores = all_scores[nonempty_index]

        CLASSES = ['creatures', 'fish', 'jellyfish', 'penguin',
                   'puffin', 'shark', 'starfish', 'stingray']
        # plot
        # https://blog.csdn.net/jyl1999xxxx/article/details/78737885
        import cv2
        report_img = cv2.imread(report_img_name)
        # color = [(0,0,255), (0,255,0), (255,0,0),
        #          (0,0,255), (0,0,255), (0,0,255),
        #          (0,0,255), (0,0,255)]
        for i, _ in enumerate(all_boxes):
            xmin, ymin, xmax, ymax = all_boxes[i]
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            cls = all_labels[i]
            score = all_scores[i]
            # print((xmin, xmax))
            # print((ymin, ymax))
            cv2.rectangle(report_img,(xmin, ymin),(xmax, ymax), (0,255,0), 4)
            font = cv2.FONT_HERSHEY_COMPLEX
            text = f'{CLASSES[cls]}: {score:0.2f}'
            cv2.putText(report_img, text, (xmin, ymin), font, 1, (0,255,0), 2)
        cv2.imwrite("./report.jpg", report_img)
        return
    

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

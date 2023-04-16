# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator

import numpy as np
import torch.nn.functional as F

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        # print(outputs['pred_logits'].shape)
        # print("out",torch.argmax(outputs['pred_logits'],axis=2)[0])
        # print("target",targets[0]['labels'])

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        # print(len(outputs['pred_logits'][0][0])) # outputs['pred_logits']: (bs, 100 possible bbox, 8+1 classes)
        # print(len(outputs['pred_boxes'][0][0])) # outputs['pred_boxes']: (bs, 100 possible bbox, (x_min, y_min, x_max, y_max))
        #print(outputs['pred_boxes'][0][0])
        #print(targets[0].keys()) # ['boxes', 'labels', 'image_id', 'area', 'iscrowd', 'orig_size', 'size']
        #print(targets[0]['boxes'])
        # print(outputs['pred_logits'][0].shape)

        # print(torch.argmax(outputs['pred_logits'][0], axis=1))
        # print(targets[0]['labels'])
        # print(outputs['pred_logits'][0][0])
        # print(F.softmax(outputs['pred_logits'][0]).shape)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator

from util.box_ops import box_cxcywh_to_xyxy

@torch.no_grad()
def test(model, data_loader, device):
    model.eval()
    results = {}
    
    for i, (img, fname) in enumerate(data_loader):
        img = img.to(device)
        # print(img[0].shape)
        h, w = img[0].shape[-2:]
        # print(h,w)
        outputs = model(img) 
        outputs = {k: v.detach().cpu() for k, v in outputs.items()}
        # outputs['pred_logits']: (bs, 100 possible bbox, 8+1 classes), 
        # outputs['pred_boxes']: (bs, 100 possible bbox, (x_min, y_min, x_max, y_max))


        # "batch_size in test mode should be 1." 
        # print(outputs['pred_logits'][0].shape)
        pred_logits = F.softmax(outputs['pred_logits'][0], dim=1) # softmax
        # print(outputs['pred_boxes'][0])
        # pred_boxes = ((outputs['pred_boxes'][0] * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]) 
        # pred_boxes = box_cxcywh_to_xyxy(outputs['pred_boxes'][0])

        # (center_x, center_y, height, width) to (x_min, y_min, x_max, y_max)
        pred_boxes = box_cxcywh_to_xyxy(outputs['pred_boxes'][0] * torch.tensor([w, h, w, h], dtype=torch.float32))
        # print(pred_boxes)

        # from ("pred_logits", "pred_boxes") to ("boxes", "labels", "scores")
        res = {}
        
        all_boxes = pred_boxes.numpy() #.tolist()
        all_labels = torch.argmax(pred_logits, axis=1).numpy() #.tolist()
        all_scores = torch.amax(pred_logits, axis=1).numpy() #.tolist()
        
        nonempty_index = np.where(all_labels!=8)
        # print(nonempty_index)
        # labels = all_labels[nonempty_index]
        # scores = all_scores[nonempty_index]
        # boxes = all_boxes[nonempty_index]
        # print(all_labels)
        # print(labels.shape, scores.shape, boxes.shape)

        res['boxes'] = all_boxes[nonempty_index].tolist()
        res['labels'] = all_labels[nonempty_index].tolist()
        res['scores'] = all_scores[nonempty_index].tolist()
                
        #print(res['boxes'].shape, res['labels'].shape, res['scores'].shape)

        # batch size must be 1.  
        # print(fname)
        results[fname[0]] = res

    return results
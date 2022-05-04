"""
Train and eval functions used in main.py
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import math
import os
import sys
from typing import Iterable
from util.misc import interpolate
import torch
import numpy as np;
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, logger, max_norm: float = 0):

    model.train()
    criterion.train()
    map = mAP_()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    #metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    iter = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        iter += 1
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        map.update(outputs, targets, iter)
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

        #metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(loss=loss_value)
        #metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(map = map.calculate_mAP())

    ## Print and save mAP, APs, mIoU
    map.print_and_save(logger, 'train')
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def val_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable,
                    device: torch.device, logger, max_norm: float = 0):
    model.eval()
    map = mAP_()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = '[valid]'
    print_freq = 10
    iter = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        iter +=1
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        map.update(outputs, targets, iter)
        #metric_logger.update(map = map.calculate_mAP())

    ## Print and save mAP, APs, mIoU
    map.print_and_save(logger, 'validation')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class mAP_():
    def __init__(self):
        self.AP_dict = {}
        for thr in range(50, 100, 5):
            self.AP_dict[thr] = []
        self.IoU = []
        self.eps = 1e-05
    
    def update(self,outputs, targets, iter):
        #import pdb; pdb.set_trace()
        targets = targets[0]['masks'].detach().cpu().numpy()
        targets = targets.astype(int)
        # outputs = outputs['pred_masks']
        # outputs = outputs[:,:360:10,:,:]
        # outputs = interpolate(outputs, size=targets.shape[-2:], mode="bilinear", align_corners=False).squeeze(0).detach().cpu().numpy()
        outputs = outputs.squeeze(1)
        outputs = outputs.detach().cpu().numpy()
        outputs = outputs > 0.5
        outputs = outputs.astype(int)
        add = targets + outputs
        sub = outputs - targets 
        #import pdb; pdb.set_trace()
        TP = (add == 2).sum()#; print(TP)
        FP = (add == 1).sum()#; print(FP)
        TN = (add == 0).sum()#; print(TN)
        FN = (sub == 1).sum()#; print(FN)
        IoU = (TP / (TP+FP+FN+self.eps)) * 100
        self.IoU.append(IoU)

        if iter % 500 == 0:
            print('TP: {}, FP: {}, FN: {}, TN: {}, IoU: {}'.format(TP, FP, FN, TN, IoU))
        
        for thr in self.AP_dict.keys():
            if IoU > thr:
                self.AP_dict[thr].append(1)
            else:
                self.AP_dict[thr].append(0)

    def calculate_mIoU(self):
        num = len(self.IoU)
        total = sum(self.IoU)
        mIoU = total/num
        return mIoU

    def calculate_mAP(self):
        TP_num = 0
        total = 0
        for thr in self.AP_dict.keys():
            total += len(self.AP_dict[thr])
            TP_num += sum(self.AP_dict[thr])
        
        mAP = TP_num / total
        return mAP

    def calculate_APs(self):
        AP_dict = {}
        for thr in self.AP_dict.keys():
            total = 0; TP_num = 0
            total += len(self.AP_dict[thr])
            TP_num += sum(self.AP_dict[thr])
            AP = TP_num / total
            AP_dict[thr] = AP

        return AP_dict

    def print_and_save(self, logger, mode):
        #import pdb; pdb.set_trace()
        logger.info("========== {} Performance ==========".format(mode))
        mIoU = self.calculate_mIoU()
        logger.info('mIoU: {}'.format(mIoU))
        mAP = self.calculate_mAP()
        logger.info('mAP: {}'.format(mAP))
        ap_dict = self.calculate_APs()
        for thr in ap_dict.keys():
            logger.info('AP_'+str(thr)+': {}'.format(ap_dict[thr]))
        logger.info("====================================")


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



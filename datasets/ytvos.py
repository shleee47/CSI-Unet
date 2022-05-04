"""
YoutubeVIS data loader
"""
from pathlib import Path
from tkinter import Frame

import torch
import torch.utils.data
import torchvision
from pycocotools.ytvos import YTVOS
from pycocotools.ytvoseval import YTVOSeval
import datasets.transforms as T
from pycocotools import mask as coco_mask
import os
from PIL import Image
from random import randint
import cv2
import random
import scipy.io
import pickle
import numpy as np

class YTVOSDataset:
    def __init__(self, img_folder, ann_file, transforms, return_masks, num_frames):
        self.img_folder = img_folder
        self.ann_file = ann_file
        self._transforms = transforms
        self.return_masks = return_masks
        self.num_frames = num_frames
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.ytvos = YTVOS(ann_file)
        self.cat_ids = self.ytvos.getCatIds()
        self.vid_ids = self.ytvos.getVidIds()
        self.vid_infos = []
        for i in self.vid_ids:
            info = self.ytvos.loadVids([i])[0]
            info['filenames'] = info['file_names']
            self.vid_infos.append(info)
        self.img_ids = []
        for idx, vid_info in enumerate(self.vid_infos):
            for frame_id in range(len(vid_info['filenames'])):
                self.img_ids.append((idx, frame_id))
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        vid,  frame_id = self.img_ids[idx]
        vid_id = self.vid_infos[vid]['id']
        img = []
        vid_len = len(self.vid_infos[vid]['file_names'])
        inds = list(range(self.num_frames))
        inds = [i%vid_len for i in inds][::-1]

        for j in range(self.num_frames):
            img_path = os.path.join(str(self.img_folder), self.vid_infos[vid]['file_names'][frame_id-inds[j]])
            img.append(Image.open(img_path).convert('RGB'))
        ann_ids = self.ytvos.getAnnIds(vidIds=[vid_id])
        target = self.ytvos.loadAnns(ann_ids)
        target = {'image_id': idx, 'video_id': vid, 'frame_id': frame_id, 'annotations': target}
        target = self.prepare(img[0], target, inds, self.num_frames)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return torch.cat(img,dim=0), target

class CSIDataset:
    def __init__(self, img_folder, transforms, return_masks, num_frames):
        self.csi_folder = img_folder
        self.transforms = transforms
        self.return_masks = return_masks
        self.num_frames = num_frames
        self.prepare = ConvertCocoPolysToMask(return_masks) # csi_folder, csi, label_folder, label
        self.tile_size = (75, 135)
        self.time_step = 3
        self.csi_infos = []
        self.label_infos = []

        temp_csi = np.zeros((3, 3, 150))
        temp_label = np.zeros((300, 540))

        csi_folder_ids = os.listdir(img_folder[0]) # 27개 csi 폴더 리스트
        csi_folder_ids.sort()
        for i in csi_folder_ids:
            csi_ids = os.listdir(os.path.join(img_folder[0], i))
            csi_ids.sort(key=lambda x:int(x.split('_')[-1].split('.')[0]))

            for j in csi_ids: # mat 파일 하나씩 iteration
                csi_path = os.path.join(img_folder[0], i, j) # csi 파일 경로
                csi_info = scipy.io.loadmat(csi_path)['csi'] # csi array load
                self.csi_infos.append(csi_info) # csi_info 리스트에 모든 csi append

                try:
                    #import pdb; pdb.set_trace()
                    label_path = os.path.join(img_folder[1], i, j)
                    with open(label_path.split('.')[0]+'.pkl', 'rb') as f:
                        label_info = pickle.load(f)  
                        if label_info.ndim != 2: # 사람 여러명 있는 label을 2차원으로 합치기
                            label_info = np.sum(label_info, axis=0) # 어레이 합치기 [3, 300, 540] -> [300, 540]
                        self.label_infos.append(label_info)
                
                except:
                    #import pdb; pdb.set_trace()
                    self.label_infos.append(temp_label)
            
            # padding: video 사이사이 32로 패딩
            # for _ in range(self.num_frames):
            #     self.csi_infos.append(temp_csi)
            #     self.label_infos.append(temp_label)

        self.len = len(self.csi_infos) - self.num_frames + 1

    def __len__(self):
        # return self.len // 36
        return self.len

    def __getitem__(self, idx):
        # idx = idx*36
        csi_list = list()
        label_list = list()
       
        csi_list.append(self.csi_infos[idx])
        label_list.append(self.label_infos[idx])
        csi_array = np.array(csi_list) # 36, 3, 3, 150
        csi_array = csi_array.reshape(1, -1, 150)
        #csi_array = np.tile(csi_array, (100, 2)).reshape(-1, 300, 300)
        #csi_array = self.multi_scale(csi_array).reshape(-1, 75, 135)
        label_array = np.array(label_list)
        label_array = label_array > 0
        
        target = {'masks': torch.Tensor(label_array), 'image_id': torch.Tensor(idx)}

        return torch.Tensor(csi_array), target

    def multi_scale(self, XX):
        
        XX = XX.transpose(1,2,3,0)
        #scaled_total = np.zeros((36,3,300,540))
        #scaled_total = np.zeros((36,3,75,135))
        scaled_total = np.zeros((1,self.time_step,75,135))
        #for scale_num in range(1,4):
        for scale_num in [3]:
            tx, rx, T, _ = XX.shape ## = 36

            t = np.random.randint(T - self.time_step)
            XX_time_step = XX[:, :, t:t + self.time_step:, :]
            # XX_time_step = XX

            tx_rand = np.random.randint(tx - scale_num + 1)
            rx_rand = np.random.randint(rx - scale_num + 1)
            XX_multi_scaled = XX_time_step[tx_rand : tx_rand + scale_num, rx_rand : rx_rand + scale_num, :, :]
            XX_multi_scaled = np.tile(XX_multi_scaled, (int(np.ceil(self.tile_size[0]/scale_num)), int(np.ceil(self.tile_size[1]/scale_num)), 1, 1))[:self.tile_size[0], :self.tile_size[1], :, :]
            XX_multi_scaled = XX_multi_scaled.transpose(3, 2, 0, 1)
            scaled_total += XX_multi_scaled

        return scaled_total

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        if not polygons:
            mask = torch.zeros((height,width), dtype=torch.uint8)
        else:
            rles = coco_mask.frPyObjects(polygons, height, width)
            mask = coco_mask.decode(rles)
            if len(mask.shape) < 3:
                mask = mask[..., None]
            mask = torch.as_tensor(mask, dtype=torch.uint8)
            mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target, inds, num_frames):
        w, h = image.size
        image_id = target["image_id"]
        frame_id = target['frame_id']
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        boxes = []
        classes = []
        segmentations = []
        area = []
        iscrowd = []
        valid = []
        # add valid flag for bboxes
        for i, ann in enumerate(anno):
            for j in range(num_frames):
                bbox = ann['bboxes'][frame_id-inds[j]]
                areas = ann['areas'][frame_id-inds[j]]
                segm = ann['segmentations'][frame_id-inds[j]]
                clas = ann["category_id"]
                # for empty boxes
                if bbox is None:
                    bbox = [0,0,0,0]
                    areas = 0
                    valid.append(0)
                    clas = 0
                else:
                    valid.append(1)
                crowd = ann["iscrowd"] if "iscrowd" in ann else 0
                boxes.append(bbox)
                area.append(areas)
                segmentations.append(segm)
                classes.append(clas)
                iscrowd.append(crowd)
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        classes = torch.tensor(classes, dtype=torch.int64)
        if self.return_masks:
            masks = convert_coco_poly_to_mask(segmentations, h, w)
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor(area) 
        iscrowd = torch.tensor(iscrowd)
        target["valid"] = torch.tensor(valid)
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        return  target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResize(scales, max_size=800),
            T.PhotometricDistort(),
            T.Compose([
                     T.RandomResize([400, 500, 600]),
                     T.RandomSizeCrop(384, 600),
                     # To suit the GPU memory the scale might be different
                     T.RandomResize([300], max_size=540),#for r50
                     #T.RandomResize([280], max_size=504),#for r101
            ]),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([360], max_size=640),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.ytvos_path)
    assert root.exists(), f'provided YTVOS path {root} does not exist'
    mode = 'instances'
    # PATHS = {
    #     "train": (root / "train/JPEGImages", root / "annotations" / f'{mode}_train_sub.json'),
    #     "val": (root / "valid/JPEGImages", root / "annotations" / f'{mode}_val_sub.json'),
    # }
    PATHS = {
        "train": (root / "train/csi", root / "train/label"),
        "val": (root / "val/csi", root / "val/label")
        #"train": (root / "train/csi", root / "label" / f'{mode}_train_sub.json'),
        # "val": (root / "valid/JPEGImages", root / "annotations" / f'{mode}_val_sub.json'),
    }
    #import pdb; pdb.set_trace()
    img_folder = PATHS[image_set]
    #img_folder, ann_file = PATHS[image_set]
    #dataset = YTVOSDataset(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks, num_frames = args.num_frames)
    dataset = CSIDataset(img_folder, transforms=make_coco_transforms(image_set), return_masks=args.masks, num_frames = args.num_frames)
    return dataset

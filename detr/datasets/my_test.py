from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

import datasets.transforms as myT
from torchvision import transforms
import os
from torch.utils.data import Dataset

import glob
from PIL import Image

class MyImage(Dataset):
    def __init__(self, img_folder, transforms):
        self.transforms = transforms
        self.root = img_folder
        self.fnames = sorted(glob.glob(os.path.join(img_folder, "*.jpg")))
        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = Image.open(fname).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img, fname.split("/")[-1]
    
    def __len__(self):
        return self.num_samples
    
    
def build(folder_name, args): 

    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    img_folder = os.path.join(root, folder_name)

    tf = transforms.Compose([transforms.ToTensor(), 
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    dataset = MyImage(img_folder, transforms=tf)

    return dataset
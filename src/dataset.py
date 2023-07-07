import cv2
import numpy as np
import os
import torch
from glob import glob
from PIL import Image
from torch.utils.data import Dataset

class CarvanaDataset(Dataset):
    
    def __init__(self, data_folder, mode="train", data_limit=None, transforms=None):
        self.transforms = transforms
        self.data_folder = data_folder
        self.all_images = sorted(glob(f"{data_folder}/train/*.jpg"))
        self.all_masks = sorted(glob(f"{data_folder}/train_masks/*.gif"))
        self.train_val_split = self.get_split()
        self.mode = mode
        if data_limit:
            self.train_val_split["train"]["images"] = self.train_val_split["train"]["images"][:data_limit]
            self.train_val_split["train"]["masks"] = self.train_val_split["train"]["masks"][:data_limit]
            self.train_val_split["val"]["images"] = self.train_val_split["val"]["images"][:data_limit]
            self.train_val_split["val"]["masks"] = self.train_val_split["val"]["masks"][:data_limit]
        self.images, self.masks = self.train_val_split[mode]["images"], self.train_val_split[mode]["masks"]
        self.transforms = transforms
        
    def get_split(self):
        filenames = [os.path.basename(x) for x in self.all_images]
        names = list(set([x.split('_')[0] for x in filenames]))
        train_len = int(len(names) * 0.7)
        train_val_dict = {"train": {
                            "images": [filepath for filepath in self.all_images 
                                       for name in names[:train_len] if name in filepath],
                            "masks": [filepath for filepath in self.all_masks
                                      for name in names[:train_len] if name in filepath]},
                          "val": {
                            "images": [filepath for filepath in self.all_images
                                       for name in names[train_len:] if name in filepath],
                            "masks": [filepath for filepath in self.all_masks
                                      for name in names[train_len:] if name in filepath]}
                        }
        return train_val_dict
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index: int):
        img_path = self.images[index]
        img_name = os.path.basename(img_path)[:-len(".jpg")]
        img = cv2.imread(img_path)
        mask_path = os.path.join(self.data_folder, "train_masks",
                                 f"{img_name}_mask.gif")
        mask = np.array(Image.open(mask_path))
        if self.transforms:
            transformed = self.transforms[self.mode]({'image': img, 'target': mask}, seed=45)
            return transformed['image'], transformed['target']
        else:
            return img, mask










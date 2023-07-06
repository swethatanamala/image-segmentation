import cv2
import os
import torch
from glob import glob
from torch.utils.data import Dataset

class CarvanaDataset(Dataset):
    
    def __init__(self, data_folder, mode="train", transforms=None):
        self.transforms = transforms
        self.datafolder = data_folder
        self.all_images = glob.glob(f"{data_folder}/train/*.jpg")
        self.mode = mode
        self.images, self.masks = self.select_mode_dataset()
        self.transforms = transforms
        
    def select_mode_dataset(self):
        length = self.__len__()
        train_length = int(0.7 * length)
        if self.mode == "train":
            images = self.all_images[:train_length]
            masks = [os.path.join(self.data_folder, "train_masks", 
                                  os.path.basename(name)[:-len(".jpg")] + "_mask.gif")
                     for name in images]
        else:
            images = self.all_images[train_length:]
            masks = [os.path.join(self.data_folder, "train_masks", 
                                  os.path.basename(name)[:-len(".jpg")] + "_mask.gif")
                     for name in images]
        return images, masks
            
    def __len__(self):
        image_names = [os.path.basename(name)[:-len(".jpg")] for name in self.images]
        mask_names = [os.path.basename(name)[:-len("_mask.gif")] for name in self.masks]
        return len(set(image_names) & set(mask_names))
    
    def __getitem__(self, index: int):
        img_path = self.images[index]
        img_name = os.path.basename(img_path)[:-len(".jpg")]
        img = cv2.imread(img_name)
        mask_path = os.path.join(self.data_folder, "train_masks",
                                 f"{img_name}_mask.gif")
        mask = cv2.imread(mask_path, 0)
        if self.transforms:
            img, mask = self.transforms[self.mode](img), self.transforms[self.mode](mask)
        return img, mask










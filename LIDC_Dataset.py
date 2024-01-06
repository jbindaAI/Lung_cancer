################ CLASS CREATING TRAINING/VAL/TEST PyTorch DATASETS
from torch.utils.data import Dataset
import pandas as pd
import torch
import random
import numpy as np


class LIDC_Dataset(Dataset):
    def __init__(self, 
                data_dir='dataset',
                train_mode = True,
                apply_mask = False,
                transform = None):
        
        self.data_dir = data_dir
        crop_path = f"{data_dir}/crops"
        mask_path = f"{data_dir}/masks"
        
        df = pd.read_pickle(f"{data_dir}/ALL_annotations_df.pkl")
        self.targets = df['target']

        imgs = []
        masks = []
        for i in range(len(self.targets)):
            imgs.append(torch.load(f"{crop_path}/{df['path'][i]}").float())
            masks.append(torch.load(f"{mask_path}/{df['path'][i]}").float())
        
        self.images = imgs
        self.masks = masks
        self.views = ["axial", "coronal", "sagittal"]
        
        # Other class attributes:
        self.train_mode = train_mode
        self.apply_mask = apply_mask
        self.transform = transform
    
    def __len__(self):
        return len(self.targets)
    
    
    def process_image(self, nodule_idx, view, slice_=16):
        # Firstly img, mask are 3D volume:
        img = self.images[nodule_idx]
        mask = self.masks[nodule_idx]
        
        # Then, I extract slice of the volume
        # at the specified view:
        if view == self.views[0]:
            img = img[:, :, slice_]
            mask = mask[:, :, slice_]
            
        elif view == self.views[1]:
            img = img[:, slice_, :]
            mask = mask[:, slice_, :]
            
        elif view == self.views[2]:
            img = img[slice_, :, :]
            mask = mask[slice_, :, :]
        
        img = torch.clamp(img, -1000, 400)
        
        # Manually adding channel dimmension:
        img = img.unsqueeze(0)
        mask = mask.unsqueeze(0)
        
        assert img.shape == mask.shape, "Image shape is not equal to mask shape!"
        
        # Rescalling pixel values to range [0, 1]:
        img -= -1000
        img = img/1400
        
        if self.apply_mask:
            img = img*mask
        
        # As ResNet model requires 3 color channels,
        # code below makes 2 more channels by coping original channel.
        img = img.repeat(3,1,1)
        
        # If some image transformations are specified:
        if self.transform is not None:
            img = self.transform(img)
        
        return img.float(), mask.float()
    
    
    def __getitem__(self, idx):
        label = self.targets[idx]
        
        # For training dataset each nodule is representent by one view: axial, coronal, sagittal.
        if self.train_mode:
            view = random.choice(self.views)
            slices = np.linspace(14, 18, 5).astype(int)
            slice_ = random.choice(slices)
            img, _ = self.process_image(nodule_idx=idx, view=view, slice_=slice_) # I don't need to return mask!
            return [img, label]
            
        else:
            # for testing dataset I take all views of a nodule
            images = []
            # masks = []
            for view in self.views:
                img, _ = self.process_image(nodule_idx=idx, view=view, slice_=16) # I don't need to return mask!
                images.append(img)
                # masks.append(mask)
            return [images, label]
    
    
    def get_target(self, idx):
        target = self.targets[idx]
        return target
        
    
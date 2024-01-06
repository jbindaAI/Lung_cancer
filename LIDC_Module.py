# packages
import pytorch_lightning as pl
import torch
from LIDC_Dataset import LIDC_Dataset
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms
from util.MyRotation import MyRotation
from torch.utils.data import DataLoader
import numpy as np


####### Ligthning Data Module CLASS

class LIDCModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir="dataset",
                 fold=0,
                 apply_mask=False,
                 batch_size=32,
                 num_workers=8
                ):
        super().__init__()
        self.data_dir = data_dir
        self.fold = fold
        self.apply_mask = apply_mask
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        
    def setup(self, stage=None):
        # full dataset class initialization:
        full_dataset = LIDC_Dataset(data_dir=self.data_dir, 
                                    train_mode=False)
        
        # number of examples in the full dataset and their idx:
        num_full = len(full_dataset)
        indices_full = list(range(num_full))
        
        # all labels in the full dataset:
        all_labels = np.array([full_dataset.get_target(i) for i in range(num_full)])
        
        # Using K-Fold Stratified Cross Validation
        # to split dataset into K=5 folds.
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        gen_splits = skf.split(indices_full, all_labels)
        
        # Creating lists with idx for train and test folds:
        train_idx_folds = []
        test_idx_folds = []
        for train_idx, test_idx in gen_splits:
            # gen_splits object is iterable object of folds
            # consisted of pairs: train_idx, test_idx
            train_idx_folds.append(train_idx)
            test_idx_folds.append(test_idx)
        
        # choosing one fold:
        train_idx = train_idx_folds[self.fold]
        test_idx = test_idx_folds[self.fold]
        
        # Defining data transformations:
        channels_mean = [0.485, 0.456, 0.406]
        channels_std = [0.229, 0.224, 0.225]
        
        train_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=channels_mean, std=channels_std),
                MyRotation([0, 90, 180, 270])
            ]
        )
        
        test_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=channels_mean, std=channels_std)
            ]
        )
        
        train = LIDC_Dataset(
            data_dir=self.data_dir,
            train_mode=True,
            transform=train_transform,
            apply_mask=self.apply_mask
        )
        
        test = LIDC_Dataset(
            data_dir=self.data_dir,
            train_mode=False,
            transform=test_transform,
            apply_mask=self.apply_mask
        )
        
        self.train_data = torch.utils.data.Subset(train, train_idx)
        self.val_data = torch.utils.data.Subset(test, test_idx)
        self.test_data = torch.utils.data.Subset(test, test_idx)

    def train_dataloader(self):
        train_loader = DataLoader(self.train_data,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.num_workers,
                                  pin_memory=True
                                 )
        return train_loader
    

    def val_dataloader(self):
        val_loader = DataLoader(self.val_data,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                pin_memory=True
                               )
        return val_loader
    
    
    def test_dataloader(self):
        test_loader = DataLoader(self.test_data, batch_size=self.batch_size,
                                 num_workers=self.num_workers,
                                 pin_memory=True
                                )
        return test_loader
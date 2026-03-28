# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 16:04:55 2024

@author: Mateo-drr
"""
from typing import Literal

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch

class CustomDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data = [torch.rand(8,16)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):    
        data = self.data[idx]
        data = torch.tensor(data)
        return {
            'data':data,
        }

def make_train_dl(config, split: Literal["train", "valid", "test"]):
    dataset = CustomDataset(config)
    dataloader = None
    match split:
        case "train":
            dataloader = DataLoader(
                dataset,
                batch_size=config.batch,
                pin_memory=True,
                shuffle=True,
                num_workers=config.num_workers,
                prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
            )
        case "valid":
            dataloader = DataLoader(
                dataset,
                batch_size=config.batch,
                pin_memory=True,
                shuffle=False,
                num_workers=config.num_workers,
            )
        case "test":
            dataloader = DataLoader(
                dataset,
                batch_size=config.batch,
                pin_memory=True,
                shuffle=False,
                num_workers=config.num_workers,
            )
        case _:
            raise NotImplementedError

    return dataloader
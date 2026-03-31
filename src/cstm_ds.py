# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 16:04:55 2024

@author: Mateo-drr
"""
from typing import Literal

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import pandas as pd
import ast

class CustomDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.train_df = pd.read_csv(config.dsPath / "train.csv")
        self.soundscapes_df = pd.read_csv(config.dsPath / "train_soundscapes.csv")

        train_audio_paths = list((config.dsPath / "train_audio").rglob("*.ogg"))
        soundscapes_audio_paths = list((config.dsPath / "train_soundscapes").rglob("*.ogg"))

        # filenames here are subfolder/name.ogg
        train_dict_map = {
            f"{file.parent.name}/{file.name}": file for file in train_audio_paths
        }
        # filenames here are just name.ogg
        sc_dict_map = {
            file.name : file for file in soundscapes_audio_paths
        }

        # one file can have many labels for the same window
        clean = {}
        for _, row in self.soundscapes_df.iterrows():
            key = (row["filename"], row["start"], row["end"])
            # get the labels in that row
            labels = [l.strip() for l in row["primary_label"].split(";")]
            if key not in clean:
                clean[key] = set()
            clean[key].update(labels)  # add any missing labels to the file window set
        # convert to dataframe
        rows = []
        for (filename, start, end), labels in clean.items():
            # take advantage of loop to replace file name with whole ds path object
            rows.append({
                "filename": sc_dict_map[filename],
                "start": start,
                "end": end,
                "label_list": sorted(labels),
                "lat_lon": None,
                "rating": None
            })

        self.soundscapes_clean = pd.DataFrame(rows)

        rows = []
        for _, row in self.train_df.iterrows():
            rows.append({
                "filename": train_dict_map[row["filename"]],
                "start": None,
                "end": None,
                # second label is a list written as str, no sec lbl are []
                "label_list": [row["primary_label"]] + ast.literal_eval(row["secondary_labels"]),
                "lat_lon": (row["latitude"], row["longitude"]),
                "rating": row["rating"],
            })

        self.train_clean = pd.DataFrame(rows)

        self.data = pd.concat(
            [self.train_clean, self.soundscapes_clean], ignore_index=True
        )

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
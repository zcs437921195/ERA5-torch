import os
import torch
from torch.utils.data import TensorDataset, DataLoader

import sys 
sys.path.append(os.path.dirname(__file__))
from Datasets import ERA5


def build_loader(config):
    train_datasets = build_dataset(config, is_train=True)
    train_loader = DataLoader(train_datasets, batch_size=config.BATCH_SIZE, shuffle=True)
    valid_datasets = build_dataset(config, is_train=False)
    valid_loader = DataLoader(valid_datasets, batch_size=config.BATCH_SIZE)
    return train_loader, valid_loader
    # return None, valid_loader


def build_dataset(config, is_train: bool):
    if config.DATASET == "ERA5":
        if is_train:
            ll = ERA5(config.TRAIN_CFG)
            datasets = ll.load()["data"]
            # inps = torch.tensor(datasets[:, :config.TRAIN_CFG["inp_sql_len"]], 
            #                     dtype=torch.float32,
            #                     device=config.DEVICE,
            #                     )
            # tgts = torch.tensor(datasets[:, config.TRAIN_CFG["inp_sql_len"]:], 
            #                     dtype=torch.float32,
            #                     device=config.DEVICE,
            #                     )
            e = len(config.TRAIN_CFG['elements'])
            h, w = config.TRAIN_CFG['height'], config.TRAIN_CFG['width']
            datasets = datasets.reshape((-1, e, h, w))
            inps = torch.tensor(datasets, 
                                dtype=torch.float32,
                                device=config.DEVICE,
                                )
            tgts = inps.clone()
        else:
            ll = ERA5(config.VALID_CFG)
            datasets = ll.load()["data"]
            # inps = torch.tensor(datasets[:, :config.valid_cfg["inp_sql_len"]], 
            #                     dtype=torch.float32,
            #                     device=config.DEVICE,
            #                     )
            # tgts = torch.tensor(datasets[:, config.VALID_CFG["inp_sql_len"]:], 
            #                     dtype=torch.float32,
            #                     device=config.DEVICE,
            #                     )
            e = len(config.VALID_CFG['elements'])
            h, w = config.VALID_CFG['height'], config.VALID_CFG['width']
            datasets = datasets.reshape((-1, e, h, w))
            inps = torch.tensor(datasets, 
                                dtype=torch.float32,
                                device=config.DEVICE,
                                )
            tgts = inps.clone()
        datasets = TensorDataset(inps, tgts)
        return datasets


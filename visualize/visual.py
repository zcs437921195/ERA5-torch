import os
import numpy as np

import sys 
sys.path.append(os.path.dirname(__file__))
from single_frame import VisualFrame
from dataload.Datasets import ERA5


def visual_model(config, model, data_loader, save_fig: bool):
    inps, tgts = data_loader.dataset[:]
    preds = model(inps)
    if "cuda" in config.DEVICE.type:
        preds, tgts = preds.detach().cpu().numpy(), tgts.detach().cpu().numpy()
    else:
        preds, tgts = preds.numpy(), tgts.numpy()
    preds = preds.reshape((-1, config.VALID_CFG["height"], config.VALID_CFG["width"]))
    tgts = tgts.reshape((-1, config.VALID_CFG["height"], config.VALID_CFG["width"]))
    ll = ERA5(config.VALID_CFG)
    datasets = ll.load()
    lon, lat = datasets["lon"], datasets["lat"]
    lon = np.broadcast_to(np.array(lon), (config.VALID_CFG["width"], config.VALID_CFG["width"]))
    lat = np.broadcast_to(np.array(lat), (config.VALID_CFG["height"], config.VALID_CFG["height"]))
    Visual = VisualFrame(save_path=config.VISUAL_PATH)
    Visual.visual_pcolor(lon, lat, preds, tgts)
    

    return 0
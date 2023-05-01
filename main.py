import argparse
import datetime
import torch
from torch import optim
from collections import defaultdict


import config as ori_config
from config import update_config
from dataload.build import build_loader
from model.unet import UNet
from model.vit import vit_base_patch16
from model.mae import mae_vit_base_patch16_dec512d8b
from loss import CalLoss 
from train import train_start


def parse_option():
    parser = argparse.ArgumentParser("Training and validation script")
    # config modification
    parser.add_argument('--batch_size', type=int, help="batch size for single GPU/CPU")
    parser.add_argument('--train_data_dir', type=str, help="path to training dataset")
    parser.add_argument('--valid_data_dir', type=str, help="path to validation dataset")
    parser.add_argument('--log', type=str, help="path to output log file")
    args, _ = parser.parse_known_args()
    # update config 
    config = update_config(args, ori_config)
    return config


def main(config):
    time_start = datetime.datetime.now()
    log_contents = defaultdict(list)
    if config.TRAIN:
        train_loader, valid_loader = build_loader(config)
        log_contents['Time'] = ['Load data time: %s' % (datetime.datetime.now() - time_start).__str__()]
        # model = UNet(n_channels=1, n_classes=1)
        # model = vit_base_patch16(img_size=config.TRAIN_CFG['height'], in_chans=1)
        model = mae_vit_base_patch16_dec512d8b(img_size=config.TRAIN_CFG['height'], in_chans=1)
        model.to(config.DEVICE)
        loss_cls = CalLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.LR_RATE)
        train_start(config, model, train_loader, optimizer, loss_cls, time_start, log_contents)
    return 0


if __name__ == '__main__':
    config = parse_option()
    main(config)
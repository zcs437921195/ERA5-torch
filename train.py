import torch
import time
import datetime
import os
from torch.utils.tensorboard import SummaryWriter


from utils.write_log import step_log
from utils.write_log import update_log


def train_one_epoch(config, 
                    model, 
                    train_loader, 
                    optimizer,
                    loss_cls,
                    writer=None):
    for s, t in train_loader:
        # print(s[:, 0].size())
        loss = 0.0
        y = model(s)
        loss_dict = loss_cls.loss(y, t)
        for _, scalar in loss_dict.items():
            loss += scalar
        loss_dict["total_loss"] = loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, loss_dict


def train_start(config, model, train_loader, optimizer, loss_cls, time_start, log_contents):
    if not os.path.exists(config.SUMMARY_PATH):
        os.makedirs(config.SUMMARY_PATH)
    writer = SummaryWriter(config.SUMMARY_PATH)
    log_contents['Time'].append(-1)
    for epc in range(config.EPOCH):
        model.train()
        model, loss_dict = train_one_epoch(config, model, train_loader, optimizer, loss_cls, writer)
        log_contents['Time'][-1] = 'cost time: %s' % (datetime.datetime.now() - time_start).__str__()
        # 记录log
        log_contents['Loss'] = []
        for tag, scalar in loss_dict.items():
            log_contents['Loss'].append('%s: %.6f' % (tag, scalar.float()))
            writer.add_scalar("Loss/%s" % tag, scalar, epc)
            writer.flush()
        log_contents['Epochs'] = ['%d/%d' % (epc, config.EPOCH)]
        epc_log = step_log(config, log_contents)
        print(epc_log)
        update_log(config, epc_log)
    writer.close()
    return 0
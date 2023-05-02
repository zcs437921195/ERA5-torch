import torch
import time
import datetime
import os
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter


from utils.log import update_log
from utils.log import write_log
from evaluation.eval import EvalMetric


def update_tensorboard_log(config, 
                           log_contents: dict,
                           update_name: str,
                           hori_value: int,
                           vert_dict: dict=None,
                           writer: SummaryWriter=None, 
                           ):
    log_contents[update_name] = []
    for tag, scalar in vert_dict.items():
        log_contents[update_name].append('%s: %.6f' % (tag, scalar.float()))
        if writer is not None:
            writer.add_scalar("%s/%s" % (update_name, tag), scalar, hori_value)
            writer.flush()
    return writer


def train_one_epoch(config, 
                    model, 
                    train_loader, 
                    optimizer,
                    loss_cls,
                    writer=None):
    loss_dict = defaultdict(int)
    cnt = 0.0
    for inps, tgts in train_loader:
        # print(s[:, 0].size())
        loss = 0.0
        # s = s.expand((-1, 3, -1, -1))
        # y = model(s)
        # loss_dict = loss_cls.loss(y, t)
        pred = model(inps)
        step_loss_dict = model.forward_loss(tgts, pred)
        for key, scalar in step_loss_dict.items():
            loss += scalar
            loss_dict[key] += scalar
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cnt += 1
    loss_dict = dict(map(lambda kv: (kv[0], kv[1] / cnt), loss_dict.items()))
    loss_dict["total_loss"] = sum(loss_dict.values())
    return model, loss_dict


def train_start(config, model, train_loader, valid_loader, optimizer, loss_cls, time_start, log_contents):
    if not os.path.exists(config.SUMMARY_PATH):
        os.makedirs(config.SUMMARY_PATH)
    writer = SummaryWriter(config.SUMMARY_PATH) if config.USE_TENSORBOARD else None
    log_contents['Time'].append(-1)
    evaluation = EvalMetric()
    for epc in range(config.EPOCH):
        model.train()
        model, loss_dict = train_one_epoch(config, model, train_loader, optimizer, loss_cls, writer)
        # 记录log
        log_contents['Epochs'] = ['%d/%d' % (epc, config.EPOCH)]
        log_contents['Time'][-1] = 'cost time: %s' % (datetime.datetime.now() - time_start).__str__()
        # 记录Loss
        writer = update_tensorboard_log(config, log_contents, 'Training Loss', epc, loss_dict, writer)
        # 验证集上验证数据
        metric_dict = evaluation.eval(model, valid_loader)
        # 记录Metric
        writer = update_tensorboard_log(config, log_contents, 'Evaluation', epc, metric_dict, writer)
        epc_log = update_log(config, log_contents)
        print(epc_log)
        write_log(config, epc_log)
    if config.USE_TENSORBOARD:
        writer.close()
    return 0
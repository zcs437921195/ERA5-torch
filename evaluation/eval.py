import torch
from torch import nn
from collections import defaultdict


class EvalMetric():
    def __init__(self) -> None:
        self.MSE = nn.MSELoss()


    def eval(self, model, valid_loader):
        model.eval()
        metric_dict = defaultdict(int)
        cnt = 0.0
        for inps, tgts in valid_loader:
            pred = model(inps)
            step_loss_dict = model.forward_loss(tgts, pred)
            # pred, mask = model(inps)
            # step_loss_dict = model.forward_loss(tgts, pred, mask)
            for key, scalar in step_loss_dict.items():
                metric_dict[key] += scalar
            cnt += 1
        metric_dict = dict(map(lambda kv: (kv[0], kv[1] / cnt), metric_dict.items()))
        return metric_dict
        

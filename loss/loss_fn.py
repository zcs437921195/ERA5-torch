import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class AutoEncoderLoss(nn.Module):
    def __init__(self, batch_exist: bool=True) -> None:
        super(AutoEncoderLoss, self).__init__()
        self.batch_exist = batch_exist
        self.auto_regression_fun = nn.MSELoss()


    def _cal_auto_regression(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.auto_regression_fun(preds, targets)


    def loss(self, preds, targets):
        auto_regression_loss = 0.0
        if self.batch_exist:
            batch_size = preds.shape[0]
            for batch_inx, pred in enumerate(preds):
                target = targets[batch_inx]
                auto_regression_loss += self._cal_auto_regression(pred, target)
        else:
            batch_size = 1.
            auto_regression_loss = self._cal_auto_regression(preds, targets)
        return {"auto_regression_loss": auto_regression_loss / batch_size}

    
    def forward(self, preds, targets):
        loss_dict = self.loss(preds, targets)
        total_loss = 0.0
        for _, value in loss_dict.items():
            total_loss += value
        return total_loss
        

class CalLoss(nn.Module):
    def __init__(self, batch_exist: bool=True) -> None:
        super(CalLoss, self).__init__()
        self.batch_exist = batch_exist
        self.auto_regression_fun = nn.MSELoss()

    def _cal_auto_regression(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.auto_regression_fun(preds, targets)


    def loss(self, preds, targets) -> dict:
        auto_regression_loss = 0.0
        if self.batch_exist:
            batch_size = preds.shape[0]
            for batch_inx, pred in enumerate(preds):
                target = targets[batch_inx]
                auto_regression_loss += self._cal_auto_regression(pred, target)
        else:
            batch_size = 1.
            auto_regression_loss = self._cal_auto_regression(preds, targets)
        return {"auto_regression_loss": auto_regression_loss / batch_size}

    
    def forward(self, preds, targets):
        loss_dict = self.loss(preds, targets)
        total_loss = 0.0
        for _, value in loss_dict.items():
            total_loss += value
        return total_loss
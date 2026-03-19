import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics.functional import spearman_corrcoef, pearson_corrcoef

from .nb_utils import NegativeBinomial, pearson_corr_2d


class PearsonLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, exp, pred):
        return -pearson_corrcoef(exp, pred).nan_to_num().mean()


class ST_criterion(nn.Module):
    def __init__(self, cfg):
        super(ST_criterion, self).__init__()
        self.cfg = cfg
        self.criterion = PearsonLoss()

    def forward(self, pred, exp):
        loss = self.criterion(pred, exp)
        return loss


def calculate_nb_loss(px_scale, exp, px_theta, train):
    # with torch.amp.autocast("cuda", enabled=False):
    if train:
        library = exp.sum(0)
    else:
        library = 1e4
        exp = exp / exp.sum() * library
    px_rate = px_scale.float() * library

    px = NegativeBinomial(mu=px_rate, theta=torch.exp(px_theta), scale=px_scale)
    # Supervised loss for Fusion
    loss = -px.log_prob(exp).sum(-1)
    return loss, px_rate


def calculate_pcc_loss(px_rate, exp):
    exp = exp / exp.sum()
    return -pearson_corr_2d(px_rate, exp).mean()


def calculate_reg_loss(exp_temp, exp_temp_new, version):
    if "1reg" in version:
        reg_loss = F.mse_loss(
            F.softmax(exp_temp_new, dim=1),
            F.softmax(torch.log(exp_temp.to(exp_temp_new)), dim=1),
        )
    else:
        reg_loss = F.mse_loss(
            torch.exp(exp_temp),
            exp_temp_new,
        )
    return reg_loss


def calculate_prop_loss(prop_pred, prop, version, slide_name=None):
    prop = prop / prop.sum()
    if "ema" in version:
        return torch.mean(torch.abs(prop_pred.mean(0) - prop))
    elif "mse" in version:
        return torch.mean((prop_pred.mean(0) - prop) ** 2)
    elif "mae" in version:
        return torch.mean(torch.abs(prop_pred.mean(0) - prop))
    else:
        return F.cross_entropy(prop_pred.mean(0), prop, reduction="sum")


class Bulk_criterion(nn.Module):
    def __init__(self, cfg, version="base"):
        super(Bulk_criterion, self).__init__()
        self.cfg = cfg
        self.version = version
        self.nb_loss = calculate_nb_loss
        self.pcc_loss = calculate_pcc_loss
        self.prop_loss = calculate_prop_loss
        self.reg_loss = calculate_reg_loss

    def forward(self, pred, exp, px_theta, probs_pred, probs, slide_name, exp_temp, exp_temp_new, train):
        if "pcc" in self.version:
            loss = self.pcc_loss(pred, exp)
        else:
            loss, px_rate = self.nb_loss(pred, exp, px_theta, train=train)

        # if train and ("scratch" not in self.version):
        #     if probs is not None:
        #         prop_loss = self.prop_loss(probs_pred, probs, self.version, slide_name)
        #     else:
        #         prop_loss = 0
        #     reg_loss = self.reg_loss(exp_temp, exp_temp_new, self.version)
        #     if "strong_prop" in self.version:
        #         loss = loss + self.reg_weight * (1e3 * prop_loss + reg_loss)
        #     elif "proto_reg" in self.version:
        #         loss = loss + self.pcc_loss(exp_temp, exp_temp_new)
        #     else:
        #         loss = loss + self.reg_weight.to(pred) * (prop_loss + reg_loss)

        return loss

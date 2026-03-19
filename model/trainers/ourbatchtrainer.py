from pathlib import Path
import warnings
from scvi import settings

# import wget
import numpy as np
import jax.numpy as jnp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Gamma, constraints
from torch.distributions import Poisson as PoissonTorch

from model.trainers import LightningModel
from .nb_utils import NegativeBinomial, pearson_corr_2d


def update_labels_ema(old_labels, new_labels, ema_decay=0.9):
    """
    Returns a label updated by EMA from old labels and new label estimates.

    Args:
        old_labels (torch.Tensor): Previous labels. Shape is arbitrary (e.g., (N, ...)).
        new_labels (torch.Tensor): New label estimates. Must have the same shape as old_labels.
        ema_decay (float): EMA decay coefficient (0~1). Larger values retain more influence from old labels.

    Returns:
        torch.Tensor: Label after EMA update.
    """
    # EMA update formula: updated = ema_decay * old + (1 - ema_decay) * new
    updated_labels = ema_decay * old_labels + (1 - ema_decay) * new_labels
    return updated_labels


class OurTrainer(LightningModel):
    def __init__(self, model, args, cfg):
        super().__init__(model, args, cfg)
        self.px_theta = nn.Parameter(torch.Tensor(cfg.MODEL.output_dim))
        nn.init.normal_(self.px_theta)

        self.reg_weight = args.reg_weight
        self.method = args.method
        self.version = args.version

    def forward(self, patch, coords, exp, prop=None, slide_name=None, train=True):
        px_scale, prop_pred = self.model(patch, coords)
        if "pcc" in self.version:
            loss = self.calculate_pcc_loss(px_scale, exp)
        else:
            loss, px_rate = self.calculate_nb_loss(px_scale, exp, train)

        if train and ("scratch" not in self.version):
            prop_loss = self.calculate_prop_loss(prop_pred, prop, slide_name)
            reg_loss = self.calculate_reg_loss()
            if "strong_prop" in self.version:
                loss = loss + self.reg_weight * (1e3 * prop_loss + reg_loss)

            else:
                loss = loss + self.reg_weight * (prop_loss + reg_loss)

        return loss, px_scale

    def calculate_nb_loss(self, px_scale, exp, train):
        # with torch.amp.autocast("cuda", enabled=False):
        if train:
            library = exp.sum(0)
        else:
            library = 1e4
            exp = exp / exp.sum() * library
        px_rate = px_scale.float() * library

        px = NegativeBinomial(mu=px_rate, theta=torch.exp(self.px_theta), scale=px_scale)
        # Supervised loss for Fusion
        loss = -px.log_prob(exp).sum(-1)
        return loss, px_rate

    def calculate_pcc_loss(self, px_rate, exp):
        exp = exp / exp.sum()
        return -pearson_corr_2d(px_rate, exp).mean()

    def calculate_prop_loss(self, prop_pred, prop, slide_name=None):
        prop = prop / prop.sum()
        if "ema" in self.version:
            return torch.mean(torch.abs(prop_pred.mean(0) - prop))
        elif "mse" in self.version:
            return torch.mean((prop_pred.mean(0) - prop) ** 2)
        elif "mae" in self.version:
            return torch.mean(torch.abs(prop_pred.mean(0) - prop))
        else:
            return F.cross_entropy(prop_pred.mean(0), prop, reduction="sum")

    def calculate_reg_loss(self):
        if "1reg" in self.version:
            reg_loss = F.mse_loss(
                F.softmax(self.model.exp_template, dim=1),
                F.softmax(torch.log(self.model.org_temp.to(self.model.exp_template)), dim=1),
            )
        else:
            reg_loss = F.mse_loss(
                torch.exp(self.model.exp_template),
                self.model.org_temp.to(self.model.exp_template),
            )
        return reg_loss

    def training_step(self, batch, batch_idx):
        """Train the model."""
        total_loss = 0
        pcc_loss = 0
        px_scale_list = []
        exp_list = []
        for data in batch:
            patch, exp, coords, slide_name, prop = (
                data["patch"],
                data["exp"],
                data["coords"],
                data["slide_name"],
                data["prop"],
            )

            loss, px_rate = self(patch, coords, exp, prop=prop, slide_name=slide_name, train=True)

            total_loss += loss
            px_scale_list.append(px_rate)
            exp_list.append(exp / exp.sum())
        pcc_loss += pearson_corr_2d(torch.stack(px_scale_list), torch.stack(exp_list)).mean()
        total_loss = total_loss / len(batch)
        self.log("train_loss", total_loss, sync_dist=True)
        self.log("train_pcc_loss", pcc_loss, sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx):

        patch, exp, slide_name, coords = (
            batch[0]["patch"],
            batch[0]["exp"],
            batch[0]["slide_name"],
            batch[0]["coords"],
        )

        loss, px_rate = self(patch, coords, exp, train=False)

        mse = F.mse_loss(px_rate, exp).detach().cpu()
        rmse = torch.sqrt(torch.mean((px_rate - exp) ** 2)).detach().cpu()

        pred = px_rate.detach().cpu().numpy()
        exp = (exp / exp.sum()).detach().cpu().numpy()

        self.log(
            "valid_loss",
            mse,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=1,
        )
        self.log(
            "valid_neglike",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=1,
        )

        self.validation_step_outputs.append([pred, exp, mse.numpy(), rmse.numpy(), slide_name])
        return pred

    def test_step(self, batch, batch_idx):
        """Testing the model in a sample.
        Calucate MSE, MAE and PCC for all spots in the sample.

        Returns:
            dict:
                MSE: MSE loss between pred and label
                MAE: MAE loss between pred and label
                corr: PCC between pred and label (across genes)
        """
        patch, exp, slide_name, coords = (
            batch[0]["patch"],
            batch[0]["exp"],
            batch[0]["slide_name"],
            batch[0]["coords"],
        )

        loss, px_rate = self(patch, coords, exp, train=False)

        mse = F.mse_loss(px_rate, exp).detach().cpu()
        rmse = torch.sqrt(torch.mean((px_rate - exp) ** 2)).detach().cpu()

        pred = px_rate.detach().cpu().numpy()
        exp = (exp / exp.sum()).detach().cpu().numpy()

        self.log("test_loss", loss, batch_size=1)

        self.validation_step_outputs.append([pred, exp, mse.numpy(), rmse.numpy(), slide_name])


class DeconvExp(OurTrainer):
    def __init__(self, model, args, cfg):
        super().__init__(model, args, cfg)
        self.px_theta = nn.Parameter(torch.Tensor(cfg.MODEL.output_dim))
        nn.init.normal_(self.px_theta)

        self.reg_weight = args.reg_weight
        self.method = args.method
        self.version = args.version

    def forward(self, patch, coords, exp, prop=None, slide_name=None, train=True):
        px_scale, prop_pred = self.model(patch, coords)
        if "pcc" in self.version:
            loss = self.calculate_pcc_loss(px_scale, exp)
        else:
            loss, px_rate = self.calculate_nb_loss(px_scale, exp, train)

        if train and ("scratch" not in self.version):
            prop_loss = self.calculate_prop_loss(prop_pred, prop, slide_name)
            reg_loss = self.calculate_reg_loss()
            if "strong_prop" in self.version:
                loss = loss + self.reg_weight * (1e3 * prop_loss + reg_loss)

            else:
                loss = loss + self.reg_weight * (prop_loss + reg_loss)

        return loss, px_scale


class DeconvExpST(DeconvExp):
    def test_step(self, batch, batch_idx):
        patch, exps, slide_name, coords = (
            batch[0]["patch"],
            batch[0]["exp"],
            batch[0]["slide_name"],
            batch[0]["coords"],
        )
        loss, px_rate = self(patch, coords, exp, train=False)
        preds, probs = self.model.local_estimation(patch)

        self.validation_step_outputs.append([preds, exps, 0, 0, slide_name])

import os
from pathlib import Path
import inspect
import importlib

# import wget
import numpy as np
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import torch.nn.functional as F
from einops import rearrange
from scipy.stats import pearsonr, spearmanr


# from .feat_ext import ImageEncoders


class LightningModel(pl.LightningModule):

    def __init__(self, model, args, cfg):
        super().__init__()
        # Initialize best metrics
        self.best_loss = np.inf
        self.best_cor = -1

        # Global Encoder
        self.model = model

        self.run_name = args.run_name

        self.criterion = F.mse_loss
        self.learning_rate = cfg.TRAINING.learning_rate
        self.validation_step_outputs = []

    def forward(self, wsi):
        # Global tokens
        output = self.model(wsi)
        return output

    def training_step(self, batch, batch_idx):
        """Train the model."""
        total_loss = 0
        for data in batch:
            patch, exp = (
                data["patch"],
                data["exp"],
            )
            pred_exp = self(patch)

            # Supervised loss for Fusion
            loss = self.criterion(pred_exp, exp)

            total_loss += loss
        total_loss = total_loss / len(batch)
        self.log("train_loss", total_loss, sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx):

        patch, exp, slide_name = (
            batch[0]["patch"],
            batch[0]["exp"],
            batch[0]["slide_name"],
        )
        pred = self(patch)

        mse = F.mse_loss(pred, exp).detach().cpu()
        rmse = torch.sqrt(torch.mean((pred - exp) ** 2)).detach().cpu().numpy()

        pred = pred.cpu().numpy()
        exp = exp.cpu().numpy()

        self.log(
            "valid_loss",
            mse,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=1,
        )

        self.validation_step_outputs.append([pred, exp, mse.numpy(), rmse, slide_name])
        return pred

    def on_validation_epoch_end(self):
        self.calc_performance(val=True)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        """Testing the model in a sample.
        Calucate MSE, MAE and PCC for all spots in the sample.

        Returns:
            dict:
                MSE: MSE loss between pred and label
                MAE: MAE loss between pred and label
                corr: PCC between pred and label (across genes)
        """
        patch, exp, slide_name = (
            batch[0]["patch"],
            batch[0]["exp"],
            batch[0]["slide_name"],
        )
        pred = self(patch)

        mse = F.mse_loss(pred, exp).detach().cpu()
        rmse = torch.sqrt(torch.mean((pred - exp) ** 2)).detach().cpu().numpy()

        pred = pred.cpu().numpy()
        exp = exp.cpu().numpy()

        self.log("test_loss", mse, batch_size=1)

        self.validation_step_outputs.append([pred, exp, mse.numpy(), rmse, slide_name])

    def on_test_epoch_end(self):
        self.calc_performance()
        self.validation_step_outputs.clear()

    def calc_performance(self, val=False):
        outputs = self.validation_step_outputs

        pred_list, exp_list, mse_list, rmse_list, slide_name_list = [], [], [], [], []
        for pred, exp, mse, rmse, slide_name in outputs:
            pred_list.append(pred)
            exp_list.append(exp)
            mse_list.append(mse)
            rmse_list.append(rmse)
            slide_name_list.append(slide_name)
        mse, rmse = np.stack(mse_list).mean(), np.stack(rmse_list).mean()

        preds = np.stack(pred_list)
        exps = np.stack(exp_list)

        peason_list = []
        spearman_list = []
        for pred, exp in zip(preds.T, exps.T):
            pearson_corr, _ = pearsonr(pred, exp)
            peason_list.append(pearson_corr)

            spearman_corr, _ = spearmanr(pred, exp)
            spearman_list.append(spearman_corr)

        pcc = np.nanmean(np.stack(peason_list))
        scc = np.nanmean(np.stack(spearman_list))

        if val:
            self.log("val_pcc", pcc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log("val_scc", scc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log(
                "valid_loss",
                mse,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=False,
            )

        else:
            print(pcc, scc)

            # Write results to file
            with open(f"{self.save_path}/{self.run_name}.txt", "w") as file:
                # Write header line
                file.write("mse, rmse, pcc, scc\n")
                # Write values on the second line
                file.write(f"{mse}, {rmse}, {pcc}, {scc}\n")
                # file.write(f"{mse}, {rmse}, {pcc}\n")

            np.save(
                f"{self.save_path}/prediction/{self.run_name}.npy",
                {"preds": preds, "exps": exps, "slide_name": slide_name_list},
            )
            return mse, rmse, pcc, scc

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        StepLR = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.9)
        optim_dict = {"optimizer": optim, "lr_scheduler": StepLR}
        return optim_dict

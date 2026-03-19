import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans

from model.trainers import LightningModel


class ComparisonTrainer(LightningModel):
    def __init__(self, model, args, cfg):
        super().__init__(model, args, cfg)
        self.method = args.method

        if args.method == "tRNAsformer":
            k = 49
            import torch.nn as nn

            self.head = nn.Linear(cfg.MODEL.input_dim, 1024)
        else:
            k = 100

        self.kmeans = KMeans(n_clusters=k, random_state=42)

    def forward(self, wsi):
        # Global tokens
        if self.method == "HE2RNA":
            output = self.model(wsi.permute(1, 0).unsqueeze(0))
        elif "SEQUOIA" in self.method:
            output = self.model(wsi.unsqueeze(0))
        else:
            wsi = self.head(wsi)
            output = self.model(wsi.reshape(224, 224).unsqueeze(0).unsqueeze(0))

        return output

    def training_step(self, batch, batch_idx):
        """Train the model."""
        total_loss = 0
        for data in batch:
            patch, exp, coords = (
                data["patch"],
                data["exp"],
                data["coords"],
            )
            self.kmeans.fit(coords.cpu().numpy())
            labels = self.kmeans.labels_
            patch_list = []
            for n_label in np.unique(labels):
                patch_list.append(patch[labels == n_label].mean(0))

            patch = torch.stack(patch_list)

            pred_exp = self(patch)

            # Supervised loss for Fusion
            loss = self.criterion(pred_exp, exp)

            total_loss += loss
        total_loss = total_loss / len(batch)
        self.log("train_loss", total_loss, sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx):

        patch, exp, slide_name, coords = (
            batch[0]["patch"],
            batch[0]["exp"],
            batch[0]["slide_name"],
            batch[0]["coords"],
        )

        self.kmeans.fit(coords.cpu().numpy())
        labels = self.kmeans.labels_

        patch_list = []
        for n_label in np.unique(labels):
            patch_list.append(patch[labels == n_label].mean(0))

        patch = torch.stack(patch_list)

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
        self.kmeans.fit(coords.cpu().numpy())
        labels = self.kmeans.labels_

        patch_list = []
        for n_label in np.unique(labels):
            patch_list.append(patch[labels == n_label].mean(0))

        patch = torch.stack(patch_list)

        pred = self(patch)

        mse = F.mse_loss(pred, exp).detach().cpu().numpy()
        rmse = torch.sqrt(torch.mean((pred - exp) ** 2)).detach().cpu().numpy()

        pred = pred.cpu().numpy()
        exp = exp.cpu().numpy()

        self.validation_step_outputs.append([pred, exp, mse, rmse, slide_name])

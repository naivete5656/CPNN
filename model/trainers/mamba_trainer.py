import torch
import torch.nn.functional as F

from model.trainers import LightningModel


class Mamba2DTrainer(LightningModel):
    def forward(self, wsi, coords):
        # Global tokens
        output = self.model(wsi, coords)
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
            pred_exp = self(patch, coords)

            # Supervised loss for Fusion
            loss = self.criterion(pred_exp, exp)

            total_loss += loss
        total_loss = total_loss / len(batch)
        self.log("train_loss", total_loss, sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx):

        patch, exp, coords, slide_name = (
            batch[0]["patch"],
            batch[0]["exp"],
            batch[0]["coords"],
            batch[0]["slide_name"],
        )
        pred = self(patch, coords)

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
        patch, exp, coords, slide_name = (
            batch[0]["patch"],
            batch[0]["exp"],
            batch[0]["coords"],
            batch[0]["slide_name"],
        )
        pred = self(patch, coords)

        mse = F.mse_loss(pred, exp).detach().cpu()
        rmse = torch.sqrt(torch.mean((pred - exp) ** 2)).detach().cpu().numpy()

        pred = pred.cpu().numpy()
        exp = exp.cpu().numpy()

        self.log("test_loss", mse, batch_size=1)

        self.validation_step_outputs.append([pred, exp, mse.numpy(), rmse, slide_name])

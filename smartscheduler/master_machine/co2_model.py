from ast import Dict
from lightning.pytorch import Trainer
from lightning.pytorch import LightningModule
from torch import nn
import torch.optim as optim
import torch
from smartscheduler.master_machine.tcn_pytorch import TemporalConvNet


class CO2Model(LightningModule):
    def __init__(
        self,
        tcn_hparams: Dict,
        predict_window: int,
        optimizer_name: str,
        optimizer_hparams: Dict,
    ):
        """Initialize  model

        Args:
            model_hparams (Dict): keys:  num_inputs: int, num_channels: List, kernel_size: int, dropout: float
            optimizer_name (str): Adam or SGD
            optimizer_hparams (Dict): lr: float, weight_decay: float
        """
        super().__init__()
        self.save_hyperparameters()

        self.loss_module = nn.L1Loss()

        # Has input as [batch_size, features_num, time_steps_num]
        self.tcn = TemporalConvNet(**tcn_hparams)

        embed_size = tcn_hparams["num_channels"][-1]

        self.regressor = nn.Sequential(
            nn.Linear(embed_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, predict_window),
        )

        # [batch_size, lookback_window, features_per_point, points_num]
        self.example_input_array = torch.ones(8, 24, 23, 38)

    def forward(self, x: torch.Tensor):
        batch_size, lookback, features_num, points_num = x.shape
        x = x.reshape((batch_size, features_num*points_num, lookback))
        emb = self.tcn(x)[:, :, -1]
        out = self.regressor(emb)
        return out


    def predict_step(self, batch, batch_idx):
        x, y, _ = batch
        preds = self.forward(x)
        return preds

    def training_step(self, batch, batch_idx):
        x, y, idx = batch
        preds = self.forward(x)
        loss = self.loss_module(preds, y)
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y, idx = batch
        preds = self.forward(x)
        loss = self.loss_module(preds, y)
        self.log("val_loss", loss)
        return {"loss": loss}

    def configure_optimizers(self):
        if self.hparams.optimizer_name == "Adam":
            optimizer = optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

        def lmbda(epoch):
            return 0.8**epoch

        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer, lr_lambda=lmbda
        )
        return [optimizer], [scheduler]

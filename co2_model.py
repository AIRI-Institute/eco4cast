from ast import Dict
from lightning.pytorch import Trainer
from lightning.pytorch import LightningModule
from torch import nn
import torch.optim as optim
import torch
from tcn_pytorch import TemporalConvNet
from torchsummary import summary



class ForecastingModel(nn.Module):
    def __init__(
        self,
        model_hparams: Dict,
        lookback_window: int,
        predict_window: int,
    ):
        super().__init__()

        self.tcn = TemporalConvNet(**model_hparams)
        self.regressor = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(lookback_window, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, predict_window),
        )

    def forward(self, x):
        x = x.flatten(-2, -1).swapaxes(-2, -1)
        x = self.tcn(x)  # out: [batch_size, 1, lookback_window]
        x = x.squeeze(-2)  # out: [batch_size, lookback_window]
        x = self.regressor(x)  # out: [batch_size, predict_window]
        return x


class TCNModel(LightningModule):
    def __init__(
        self,
        model_hparams: Dict,
        lookback_window: int,
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

        self.loss_module = nn.MSELoss()

        # Has input as [batch_size, features_num, time_steps_num]
        self.tcn = TemporalConvNet(**model_hparams)

        # self.downsample = nn.Conv1d(in_channels=96, out_channels=1, kernel_size=6)

        self.regressor = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(
                model_hparams["num_channels"][-1] * 8,
                1024,
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, predict_window),
        )

        # [batch_size, lookback_window, features_per_point, points_num]
        self.example_input_array = torch.ones(8, 96, 22, 40)

    def forward(self, x):
        x = x.flatten(-2, -1).swapaxes(
            -2, -1
        )  # out: [batch_size, features_num, lookback_window]
        x = self.tcn(x)  # out: [batch_size, num_channels[-1], lookback_window]
        x = x[:, :, -8:]  # out: [batch_size, num_channels[-1], 8]
        x = x.flatten(-2, -1)  # out: [batch_size, num_channels[-1]]
        x = self.regressor(x)  # out: [batch_size, predict_window]
        return x

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


if __name__ == "__main__":
    # model = TemporalConvNet(880, [128], kernel_size=3)

    # print(model(torch.ones(8, 880, 96)).shape)

    model = TCNModel(
        {
            "num_inputs": 22 * 40,
            "num_channels": [124, 124, 124, 124],
            "kernel_size": 3,
            "dropout": 0.2,
        },
        96,
        24,
        "Adam",
        {"lr": 1e-3, "weight_decay": 1e-4},
    )

    print(model(torch.ones(8, 96, 22, 40)).shape)
    print(sum(p.numel() for p in model.parameters() ))
    print(summary(model, (96, 22, 40))) 

    print(model)


    def count_parameters(model):
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            params = parameter.numel()
            total_params+=params
        print(f"Total Trainable Params: {total_params}")
        return total_params
        

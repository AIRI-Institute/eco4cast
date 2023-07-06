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
        tcn_hparams: Dict,
        lookback_window: int,
        predict_window: int,
    ):
        super().__init__()

        self.tcn = TemporalConvNet(**tcn_hparams)
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


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, ffn_hidden, num_heads, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(
            d_model=embed_dim, hidden=ffn_hidden, drop_prob=drop_prob
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x):
        # 1. compute self attention
        _x = x
        x, _= self.attention(x, x, x)

        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x


class CO2Model(LightningModule):
    def __init__(
        self,
        tcn_hparams: Dict,
        attention_layers_num : int,
        predict_window: int,
        optimizer_name: str,
        optimizer_hparams: Dict,
        layernorm_dropout=0.2,
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

        # self.encoder_layers = [EncoderLayer(embed_size, 512, 16, 0.0) for _ in range(attention_layers_num)]
        # self.encoder_layers = nn.Sequential(*self.encoder_layers)

        self.regressor = nn.Sequential(
            nn.Linear(embed_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, predict_window),
        )

        # [batch_size, lookback_window, features_per_point, points_num]
        self.example_input_array = torch.ones(8, 24, 23, 38)

    def forward(self, x: torch.Tensor):
        # batch_size, lookback, features_num, points_num = x.shape
        # x = x.reshape((batch_size * points_num, features_num, lookback))
        # embed = self.tcn(x)[:, :, -1]
        # embed = embed.reshape(batch_size, points_num, -1)

        # regression_point = torch.ones(
        #     (batch_size, 1, embed.shape[-1]), device=self.device
        # )
        # out = torch.cat((regression_point, embed), dim=1)

        # out = self.encoder_layers(out)

        # out = out[:, 0, :]
        # out = self.regressor(out)
        # return out

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


if __name__ == "__main__":
    model = CO2Model(
        tcn_hparams={
            "num_inputs": 21,
            "num_channels": [32, 64, 64, 128, 128],
            "kernel_size": 3,
            "dropout": 0.2,
        },
        lookback_window=96,
        predict_window=24,
        optimizer_name="Adam",
        optimizer_hparams={"lr": 1e-3, "weight_decay": 1e-4},
    )

    model.to('cuda:0')

    print(model(torch.ones(8, 96, 21, 40).cuda()).shape)
    print(sum(p.numel() for p in model.parameters()))
    print(summary(model, (96, 21, 40), device="cpu"))

    print(model)

 
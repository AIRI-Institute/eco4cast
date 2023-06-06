from model import TCNModel
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from dataset import WeatherCO2DataModule
import numpy as np
import torch


torch.manual_seed(0)

model = TCNModel(
    {
        "num_inputs": 22 * 40,
        "num_channels": [512, 256, 128, 128, 64, 32],
        "kernel_size": 4,
        "dropout": 0.2,
    },
    96,
    24,
    "SGD",
    {"lr": 1e-4, "weight_decay": 1e-5},
)


wandb_logger = WandbLogger(log_model="all", project="SmartScheduler")

trainer = Trainer(
    accelerator="gpu",
    devices=[0],
    max_epochs=100,
    callbacks=[
        ModelCheckpoint(
            save_weights_only=True,
            monitor="val_loss",
            mode="min",
        ),
        EarlyStopping(monitor="val_loss", mode="min", patience=20),
    ],
    default_root_dir="models",
    logger=wandb_logger,
)


weather_points_dataset = np.load("weather_data_2021-2023.npy", allow_pickle=True)
emission_data = np.load("emission_data_2021-2023.npy", allow_pickle=True)

datetimes = list(range(len(emission_data)))

dm = WeatherCO2DataModule(
    weather_points_dataset,
    emission_data,
    datetimes,
    96,
    24,
    num_workers=5,
    batch_size=32,
    split_sizes=[0.8, 0.2, 0.0],
)


trainer.fit(model, datamodule=dm)

from co2_model import CO2Model
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from weather_co2_dataset import WeatherCO2DataModule
import numpy as np
import torch
import pandas as pd


torch.manual_seed(0)

model = CO2Model(
    tcn_hparams={
        "num_inputs": 23*38,
        # "num_channels": [64, 128, 128, 128, 256],
        "num_channels": [512, 256, 256, 256, 256],
        "kernel_size": 3,
        "dropout": 0.0,
    },
    attention_layers_num = 8,
    predict_window=24,
    optimizer_name="Adam",
    optimizer_hparams={"lr": 1e-3, "weight_decay": 1e-4},
)


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
    logger=WandbLogger(log_model="all", project="SmartScheduler"),
)


codes = ["BR-CS", "CA-ON", "CH", "DE", "PL", "BE", "IT-NO", "CA-QC", "ES", "GB", "FI", "FR", "NL"]

features_data = []
targets_data = []
for code in codes:
    features = np.load(f"electricitymaps_datasets/{code}_np_dataset.npy", allow_pickle=True)
    features_data.append(features)

    emission_df = pd.concat(
        (
            pd.read_csv(f"electricitymaps_datasets/{code}_2021_hourly.csv"),
            pd.read_csv(f"electricitymaps_datasets/{code}_2022_hourly.csv"),
        )
    ).reset_index(drop=True)
    target = emission_df["Carbon Intensity gCO₂eq/kWh (LCA)"].to_numpy()
    targets_data.append(target)


dm = WeatherCO2DataModule(
    features_data, targets_data, 24, 24, 10, 64
)


trainer.fit(model, datamodule=dm)

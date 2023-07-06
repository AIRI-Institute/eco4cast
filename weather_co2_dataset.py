from ast import List
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from weather_data_utils import (
    get_points_over_country,
    get_multiple_historic_data,
    get_emission_data,
)
from lightning.pytorch import LightningDataModule
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd


class WeatherCO2Dataset(Dataset):
    def __init__(
        self,
        features_data,
        target_data,
        # datetime_data,
        lookback_window,
        predict_window,
        use_min_max_scaling=False,
        padding_to_size=38,
    ) -> None:
        super().__init__()
        # Shape : [time_steps_num, features_num, points_num]
        # self.datetime_data = datetime_data
        self.lookback_window, self.predict_window = lookback_window, predict_window

        if use_min_max_scaling:
            self.features_scaler = MinMaxScaler()
            features_data = self.features_scaler.fit_transform(
                features_data.reshape(-1, features_data.shape[1])
            ).reshape(features_data.shape)

            self.target_scaler = MinMaxScaler()
            target_data = self.target_scaler.fit_transform(
                target_data.reshape(-1, 1)
            ).squeeze()

        self.features_data = torch.tensor(features_data.astype(np.float32))
        self.target_data = torch.tensor(target_data.astype(np.float32))

        self.padding_to_size = padding_to_size

        assert len(features_data) == len(target_data)
        # assert len(target_data) == len(datetime_data)

    def __len__(self):
        return len(self.features_data) - self.lookback_window - self.predict_window + 1

    def __getitem__(self, index):
        features_history = self.features_data[index : index + self.lookback_window]
        target = self.target_data[
            index
            + self.lookback_window : index
            + self.lookback_window
            + self.predict_window
        ]
        if self.padding_to_size - features_history.shape[2] > 0:
            padding = torch.zeros(
                (
                    features_history.shape[0],
                    features_history.shape[1],
                    self.padding_to_size - features_history.shape[2],
                )
            )
            features_history = torch.cat((features_history, padding), dim=2)

        return features_history, target, index


class WeatherCO2DataModule(LightningDataModule):
    def __init__(
        self,
        all_features_data: List,
        all_target_data: List,
        lookback_window,
        predict_window,
        num_workers,
        batch_size,
        split_sizes=[0.7, 0.3],  # Train/val,
    ):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.all_features_data = all_features_data
        self.all_target_data = all_target_data

        self.lookback_window = lookback_window
        self.predict_window = predict_window
        self.split_sizes = split_sizes

    def setup(self, stage: str):
        train_datasets = []
        val_datasets = []
        for features_data, target_data in zip(
            self.all_features_data, self.all_target_data
        ):
            train_size = int(self.split_sizes[0] * len(features_data))
            val_size = int(self.split_sizes[1] * len(features_data))
            train_dataset = WeatherCO2Dataset(
                features_data=features_data[:train_size],
                target_data=target_data[:train_size],
                lookback_window=self.lookback_window,
                predict_window=self.predict_window,
            )
            val_dataset = WeatherCO2Dataset(
                features_data=features_data[train_size : train_size + val_size],
                target_data=target_data[train_size : train_size + val_size],
                lookback_window=self.lookback_window,
                predict_window=self.predict_window,
            )

            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)

        self.train_dataset = ConcatDataset(train_datasets)
        self.val_dataset = ConcatDataset(val_datasets)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )


if __name__ == "__main__":
    codes = [
        "BR-CS",
        "CA-ON",
        "CH",
        "DE",
        "PL",
        "BE",
        "IT-NO",
        "CA-QC",
        "ES",
        "GB",
        "FI",
        "FR",
        "NL",
    ]

    features_data = []
    targets_data = []
    for code in codes[:2]:
        features = np.load(
            f"electricitymaps_datasets/{code}_np_dataset.npy", allow_pickle=True
        )
        features_data.append(features)

        emission_df = pd.concat(
            (
                pd.read_csv(f"electricitymaps_datasets/{code}_2021_hourly.csv"),
                pd.read_csv(f"electricitymaps_datasets/{code}_2022_hourly.csv"),
            )
        ).reset_index(drop=True)
        target = emission_df["Carbon Intensity gCO₂eq/kWh (LCA)"].to_numpy()
        targets_data.append(target)

    dm = WeatherCO2DataModule(features_data, targets_data, 96, 24, 10, 16)

    dm.setup("fit")

    print(
        dm.train_dataset[0][0].shape,
        dm.train_dataset[0][1].shape,
        dm.train_dataset[0][0].mean(),
        dm.train_dataset[0][1].mean(),
    )

    print("Total dataset samples: ", len(dm.train_dataset) + len(dm.val_dataset))

    # joblib.dump(dm.train_dataset.features_scaler, 'features_scaler.gz')
    # joblib.dump(dm.train_dataset.target_scaler, 'target_scaler.gz')

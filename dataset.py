import torch
from torch.utils.data import Dataset, DataLoader
from weather_data_utils import (
    get_points_over_country,
    get_multiple_historic_data,
    get_emission_data,
    average_emission_data,
)
from lightning.pytorch import LightningDataModule
import numpy as np


class WeatherCO2Dataset(Dataset):
    # TODO : is it necessary to use MinMaxScaler?

    def __init__(
        self, features_data, target_data, datetime_data, lookback_window, predict_window
    ) -> None:
        super().__init__()

        self.features_data = torch.tensor(features_data.astype(np.float32))
        self.target_data = torch.tensor(target_data.astype(np.float32))
        self.datetime_data = datetime_data
        self.lookback_window, self.predict_window = lookback_window, predict_window

        assert len(features_data) == len(target_data)
        assert len(target_data) == len(datetime_data)

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

        return features_history, target, index


class WeatherCO2DataModule(LightningDataModule):
    def __init__(
        self,
        all_features_data,
        all_target_data,
        all_datetimes,
        lookback_window,
        predict_window,
        num_workers,
        batch_size,
        split_sizes=[0.6, 0.2, 0.2],
    ):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.features_data = all_features_data
        self.target_data = all_target_data
        self.datetimes = all_datetimes

        self.lookback_window = lookback_window
        self.predict_window = predict_window
        self.split_sizes = split_sizes
        # self.datetime_data = all_datetime_data

    def setup(self, stage: str):
        train_size = int(self.split_sizes[0] * len(self.features_data))
        val_size = int(self.split_sizes[1] * len(self.features_data))
        # test_size = int(self.split_sizes[2] * len(self.features_data))
        if stage == "fit":
            self.train_dataset = WeatherCO2Dataset(
                features_data=self.features_data[:train_size],
                target_data=self.target_data[:train_size],
                datetime_data=self.datetimes[:train_size],
                lookback_window=self.lookback_window,
                predict_window=self.predict_window,
            )
            self.val_dataset = WeatherCO2Dataset(
                features_data=self.features_data[train_size : train_size + val_size],
                target_data=self.target_data[train_size : train_size + val_size],
                datetime_data=self.datetimes[train_size : train_size + val_size],
                lookback_window=self.lookback_window,
                predict_window=self.predict_window,
            )
        if stage == "test":
            self.test_dataset = WeatherCO2Dataset(
                features_data=self.features_data[train_size + val_size :],
                target_data=self.target_data[train_size + val_size :],
                datetime_data=self.datetimes[train_size + val_size :],
                lookback_window=self.lookback_window,
                predict_window=self.predict_window,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )


if __name__ == "__main__":
    denmark_points = get_points_over_country("DNK", 0.5)
    assert len(denmark_points) == 40

    year_start = "2023"
    month_start = "01"
    day_start = "31"

    year_end = "2023"
    month_end = "03"
    day_end = "02"

    weather_points_dataset, datetimes, emission_df = get_multiple_historic_data(
        denmark_points,
        year_start=year_start,
        month_start=month_start,
        day_start=day_start,
        year_end=year_end,
        month_end=month_end,
        day_end=day_end,
        include_weathercode=False,
        co2_emission_delta_days=1,
    )

    emission_data = emission_df["CO2Emission"].to_numpy()

    dm = WeatherCO2DataModule(
        weather_points_dataset, emission_data, datetimes, 96, 24, 10, 16
    )

    dm.setup("fit")
    dm.setup("test")

    print("Full data: ", len(emission_data))
    print(
        "Total dataset samples: ",
        len(dm.train_dataset) + len(dm.val_dataset) + len(dm.test_dataset),
    )

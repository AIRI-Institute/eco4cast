from ast import List
from smartscheduler.master_machine.weather_data_utils import (
    get_last_weather_data,
)
from smartscheduler.master_machine.co2_model import CO2Model
import torch
import numpy as np
import datetime
from smartscheduler.master_machine.electricitymaps_api import get_24h_history
from smartscheduler.master_machine.utils import (
    codes_with_steps,
    code_names,
)
import pickle
import importlib.resources
from . import data_files
from smartscheduler.master_machine.utils import code_names


class CO2Predictor:
    """
    Class that uses TCNModel to predict co2 emission
    """

    def __init__(
        self,
        electricity_maps_api_key: str,
        checkpoint_file_path=None,
    ) -> None:
        """
        Args:
            electricity_maps_api_key : api key to access weather in electricity_maps regions
            (Optional) checkpoint_file_path (str) : path to file with torch model. If not provided - default model is used
        """

        self.lookback_window = 24  # Model-specific parameter

        self.data_path = importlib.resources.files(data_files)
        if checkpoint_file_path is None:
            checkpoint_file_path = self.data_path / "co2_model.ckpt"
        self.load_model(checkpoint_file_path=checkpoint_file_path)
        self.electricity_maps_api_key = electricity_maps_api_key

    def load_model(
        self,
        checkpoint_file_path,
    ):
        """
        This function loads model and country points for predicting time intervals.
        """
        self.predict_model = CO2Model(
            tcn_hparams={
                "num_inputs": 23 * 38,
                "num_channels": [512, 256, 256, 256, 256],
                "kernel_size": 3,
                "dropout": 0.0,
            },
            predict_window=24,
            optimizer_name="Adam",
            optimizer_hparams={"lr": 1e-3, "weight_decay": 1e-4},
        )
        self.predict_model = self.predict_model.load_from_checkpoint(
            checkpoint_file_path
        )
        self.predict_model = self.predict_model.to("cpu").eval()

        self.zones_with_steps = codes_with_steps

        with open(self.data_path / "country_points.pickle", "rb") as f:
            self.country_points = pickle.load(f)

    def predict_co2(self):
        """
        This function predicts co2 emission using model (firstly, it loads weather and emission data).
        """

        batch = []
        for code, points in self.country_points:
            point_weather_matrices = []
            zone_emission = get_24h_history(code, self.electricity_maps_api_key)
            # zone_emission = [0] * 24 # For testing purposes
            for longitude, latitude in points:
                point_weather = get_last_weather_data(
                    latitude=latitude,
                    longitude=longitude,
                    lookback_days=(self.lookback_window // 24) + 1,
                ).iloc[-self.lookback_window :]
                point_weather = point_weather.drop(columns=["weathercode", "time"])
                point_weather["emission"] = zone_emission
                point_weather["longitude"] = longitude
                point_weather["latitude"] = latitude
                point_weather = point_weather.to_numpy()
                point_weather_matrices.append(point_weather)

            for _ in range(len(points), 38):
                point_weather_matrices.append(np.zeros_like(point_weather))

            point_weather_matrices = np.array(point_weather_matrices)
            batch.append(point_weather_matrices)

        batch = np.array(batch)
        batch = torch.tensor(batch.astype(np.float32))
        batch = batch.permute((0, 2, 3, 1))

        self.emission_forecast = self.predict_model.predict_step((batch, 0, 0), 0)
        self.emission_forecast = self.emission_forecast.detach().cpu().numpy()

        return self.emission_forecast


class IntervalGenerator:
    """
    Class that uses CO2Predictor's forecast to generate intervals that satisfy conditions
    (emission in interval less than threshold)
    """

    def __init__(
        self,
        zone_names=code_names,
        max_emission_value=180,
        co2_delta_to_move=30,
        min_interval_size=1,
        max_window_size=3,
        exclude_zones=None,
        include_zones=None,
    ) -> None:
        self.zone_names = dict(zip(list(range(len(zone_names))), zone_names))
        self.zone_names[-1] = -1
        self.zone_to_id = dict(zip(zone_names, list(range(len(zone_names)))))
        self.max_emission_value = max_emission_value
        self.co2_delta_to_move = co2_delta_to_move
        self.min_interval_size = min_interval_size
        self.max_window_size = max_window_size
        self.exclude_zones = exclude_zones
        self.include_zones = include_zones

    def generate_intervals(
        self,
        forecasts,
        current_zone_idx=0,
    ):
        """
        This function generates training intervals.
        Uses co2_predictor's forecasted co2 emission to build intervals with minimum total emission.
        """

        assert len(forecasts) == len(self.zone_names) - 1

        forecasts = np.copy(forecasts)
        

        if self.include_zones is not None and len(self.include_zones) > 0:
            self.exclude_zones = [
                k for k in self.zone_to_id.keys() if k not in self.include_zones
            ]

        if self.exclude_zones is not None and len(self.exclude_zones) > 0:
            for z in self.exclude_zones:
                forecasts[self.zone_to_id[z]] = 1e9

        time_slots = np.zeros((len(forecasts), 24), dtype=int)

        for forecast_id, forecast in enumerate(forecasts):
            for window_size in range(self.max_window_size):
                for i in range(len(forecast)):
                    co2_mean = forecast[
                        i : i + self.min_interval_size + window_size
                    ].mean()
                    if co2_mean < self.max_emission_value:
                        time_slots[
                            forecast_id,
                            i : i + self.min_interval_size + window_size - 1,
                        ] = 1

        time_slots_vms = np.ones((24), dtype=int) * -1

        i = 0
        current_vm_idx = current_zone_idx

        while i + self.min_interval_size <= 24:
            if time_slots[:, i].sum() == 0:
                i += 1
                continue

            possible_vms = time_slots[:, i : i + self.min_interval_size].all(1)
            if possible_vms.sum() == 1:
                time_slots_vms[
                    i : i + self.min_interval_size
                ] = current_vm_idx = possible_vms.argmax()
                i += self.min_interval_size
            else:
                co2_mean = forecasts[:, i : i + self.min_interval_size].mean(axis=1)
                co2_mean += (~possible_vms) * 1e9
                co2_mean[current_vm_idx] -= self.co2_delta_to_move
                min_co2_idx = current_vm_idx = co2_mean.argmin()
                time_slots_vms[i : i + self.min_interval_size] = min_co2_idx
                i += self.min_interval_size

        current_vm = -1
        intervals = {}
        for time, vm in enumerate(time_slots_vms):
            if current_vm == -1:
                current_vm = vm
                start_time = time

            if vm != current_vm:
                end_time = time
                if current_vm in intervals.keys():
                    intervals[current_vm].append((start_time, end_time))
                else:
                    intervals[current_vm] = [(start_time, end_time)]
                current_vm = vm
                start_time = time

        end_time = time
        if current_vm in intervals.keys():
            intervals[current_vm].append((start_time, end_time))
        else:
            intervals[current_vm] = [(start_time, end_time)]

        if -1 in intervals.keys():
            del intervals[-1]

        ordered_intervals = [
            (k, intervals[k][i])
            for k in intervals.keys()
            for i in range(len(intervals[k]))
        ]
        ordered_intervals = sorted(ordered_intervals, key=lambda x: x[1][0])

        if len(ordered_intervals) == 0:
            return [], current_zone_idx
        formatted_intervals = [(ordered_intervals[0][0], [ordered_intervals[0][1]])]

        for vm_idx, interval in ordered_intervals[1:]:
            if formatted_intervals[-1][0] == vm_idx:
                formatted_intervals[-1][1].append(interval)
            else:
                formatted_intervals.append((vm_idx, [interval]))

        formatted_intervals

        datetime_intervals = [
            (
                self.zone_names[vm_idx],
                [
                    (
                        datetime.datetime.now(datetime.timezone.utc).replace(
                            minute=0, second=0, microsecond=0
                        )
                        + datetime.timedelta(hours=int(jstart)),
                        datetime.datetime.now(datetime.timezone.utc).replace(
                            minute=0, second=0, microsecond=0
                        )
                        + datetime.timedelta(hours=int(jend)),
                    )
                    for jstart, jend in vm_intervals
                ],
            )
            for vm_idx, vm_intervals in formatted_intervals
        ]

        return datetime_intervals, [
            vm_idx for vm_idx, vm_intervals in formatted_intervals
        ]

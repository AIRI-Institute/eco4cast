from ast import List
from smartscheduler.master_machine.weather_data_utils import (
    get_last_weather_data,
    # get_points_over_country,
)
from smartscheduler.master_machine.co2_model import CO2Model
import torch
import numpy as np
import datetime
from smartscheduler.master_machine.electricitymaps_api import get_24h_history
from smartscheduler.master_machine.utils import (
    countryISOMapping,
    codes_with_steps,
    code_names,
)
import pickle
import importlib.resources
from . import data_files

class CO2Predictor:
    """
    Class that uses TCNModel to predict co2emission
    """

    def __init__(
        self,
        checkpoint_file_path=None,
    ) -> None:
        """
        Args:
            country_code (str) : ISO-3 coutry code
            points_step (float) : step between points on weather map (in degrees)
            (Optional) checkpoint_file_path (str) : path to file with torch model
        """

        self.lookback_window = 24  # Model-specific parameter
        self.predict_window = 24  # Model-specific parameter
        self.co2_emission_delta_days = 0  # Model-specific parameter

        self.data_path = importlib.resources.files(data_files)
        if checkpoint_file_path is None:
            checkpoint_file_path = self.data_path / 'co2_model.ckpt'
        self.load_model(checkpoint_file_path=checkpoint_file_path)
        

    def load_model(
        self,
        checkpoint_file_path,
    ):
        """
        This function loads model for predicting time intervals.
        train: bool
            If 'train' is True, then model will be initialized with initial weights and then will be trained on weather and emission data.
            This is not necessary, this is totaly optional(my fantasy)
        """
        self.predict_model = CO2Model(
            tcn_hparams={
                "num_inputs": 23 * 38,
                "num_channels": [512, 256, 256, 256, 256],
                "kernel_size": 3,
                "dropout": 0.0,
            },
            attention_layers_num=8,
            predict_window=24,
            optimizer_name="Adam",
            optimizer_hparams={"lr": 1e-3, "weight_decay": 1e-4},
        )
        self.predict_model = self.predict_model.load_from_checkpoint(
            checkpoint_file_path
        )
        self.predict_model = self.predict_model.to("cpu").eval()

        self.zones_with_steps = codes_with_steps

        with open(self.data_path / 'country_points.pickle', 'rb') as f:
            self.country_points = pickle.load(f)

        # self.country_points = []
        # for code, points_step in self.zones_with_steps:
        #     points = get_points_over_country(
        #         country_code=countryISOMapping[code.split("-")[0]],
        #         points_step=points_step,
        #     )
        #     self.country_points.append((code, points))


    def predict_co2(self):
        """
        This function predicts co2 emission using model (firstly, it needs to load weather and emission data).
        """

        batch = []
        for code, points in self.country_points:
            point_weather_matrices = []
            zone_emission = get_24h_history(code)
            # zone_emission = [0] * 24
            for longitude, latitude in points:
                point_weather = get_last_weather_data(
                    latitude=10,
                    longitude=10,
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
    (emission in time_step or moving window less than threshold)
    """

    def __init__(self, zone_names=None) -> None:
        self.zone_names = dict(zip(list(range(len(zone_names))), zone_names))
        self.zone_names[-1] = -1
        self.zone_to_id = dict(zip(zone_names, list(range(len(zone_names)))))

    def generate_intervals(
        self,
        forecasts,
        current_machine=0,
        min_interval_size=1,
        max_window_size=3,
        max_emission_value=180,
        co2_delta_to_move=30,
        exclude_zones: List = [],
        include_zones: List = [],
    ):
        """
        This function generates training intervals.
        Uses co2_predictor's forecasted co2 emission to build intervals with minimum total emission.
        """

        assert len(forecasts) == len(self.zone_names) - 1

        if len(include_zones) > 0:
            exclude_zones = [
                k for k in self.zone_to_id.keys() if k not in include_zones
            ]

        if len(exclude_zones) > 0:
            for z in exclude_zones:
                forecasts[self.zone_to_id[z]] = 1e9

        time_slots = np.zeros((len(forecasts), 24), dtype=int)

        for forecast_id, forecast in enumerate(forecasts):
            for window_size in range(max_window_size):
                for i in range(len(forecast)):
                    co2_mean = forecast[i : i + min_interval_size + window_size].mean()
                    if co2_mean < max_emission_value:
                        time_slots[
                            forecast_id, i : i + min_interval_size + window_size - 1
                        ] = 1

        time_slots_vms = np.ones((24), dtype=int) * -1

        i = 0
        current_vm_idx = current_machine

        while i + min_interval_size <= 24:
            if time_slots[:, i].sum() == 0:
                i += 1
                continue

            possible_vms = time_slots[:, i : i + min_interval_size].all(1)
            if possible_vms.sum() == 1:
                time_slots_vms[
                    i : i + min_interval_size
                ] = current_vm_idx = possible_vms.argmax()
                i += min_interval_size
            else:
                co2_mean = forecasts[:, i : i + min_interval_size].mean(axis=1)
                co2_mean += (~possible_vms) * 1e9
                co2_mean[current_vm_idx] -= co2_delta_to_move
                min_co2_idx = current_vm_idx = co2_mean.argmin()
                time_slots_vms[i : i + min_interval_size] = min_co2_idx
                i += min_interval_size

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

        return datetime_intervals


if __name__ == "__main__":
    co2_predictor = CO2Predictor()
    co2_forecast = co2_predictor.predict_co2()
    print(co2_forecast)

    interval_generator = IntervalGenerator(code_names)
    print(interval_generator.generate_intervals(co2_forecast, max_emission_value=130))

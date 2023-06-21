from ast import List
from weather_data_utils import get_points_over_country, get_multiple_last_weather_data
from co2_model import TCNModel
import torch
import numpy as np
import datetime
import joblib


class CO2Predictor:
    """
    Class that uses TCNModel to predict co2emission 
    """

    def __init__(
        self,
        country_code,
        points_step,
        checkpoint_file_path=None,
    ) -> None:
        """
        Args:
            country_code (str) : ISO-3 coutry code
            points_step (float) : step between points on weather map (in degrees)
            (Optional) checkpoint_file_path (str) : path to file with torch model
        """
        self.country_points = get_points_over_country(country_code, points_step)

        self.lookback_window = 96  # Model-specific parameter
        self.predict_window = 24  # Model-specific parameter
        self.co2_emission_delta_days = 0  # Model-specific parameter

        if checkpoint_file_path is None:
            self.load_model()
        else:
            self.load_model(checkpoint_file_path=checkpoint_file_path)

    def load_model(self, train=False, checkpoint_file_path="co2_model.ckpt"):
        """
        This function loads model for predicting time intervals.
        train: bool
            If 'train' is True, then model will be initialized with initial weights and then will be trained on weather and emission data.
            This is not necessary, this is totaly optional(my fantasy)
        """
        self.predict_model = TCNModel(
            {
                "num_inputs": 22 * 40,
                "num_channels": [512, 128],
                "kernel_size": 3,
                "dropout": 0.2,
            },
            self.lookback_window,
            self.predict_window,
            "Adam",
            {"lr": 1e-3, "weight_decay": 1e-4},
        )
        self.predict_model = self.predict_model.load_from_checkpoint(
            checkpoint_file_path
        )
        self.predict_model = self.predict_model.to("cpu").eval()

        self.features_scaler = joblib.load("features_scaler.gz")
        self.target_scaler = joblib.load("target_scaler.gz")

    def predict_co2(self):
        """
        This function predicts co2 emission using model (firstly, it needs to load weather and emission data).
        """
        weather_data, weather_time, emission_df = get_multiple_last_weather_data(
            self.country_points,
            self.lookback_window,
            co2_emission_delta_days=self.co2_emission_delta_days,
        )
        real_emission_data = emission_df["CO2Emission"].to_numpy()
        weather_data = self.features_scaler.transform(
            weather_data.reshape(-1, weather_data.shape[1])
        ).reshape(weather_data.shape)

        x = torch.tensor(weather_data.astype(np.float32))
        x = x.unsqueeze(0)
        batch = (x, 0, 0)  # x, y, idx

        self.emission_forecast = self.predict_model.predict_step(batch, 0)[0]
        self.emission_forecast = self.emission_forecast.detach().cpu().numpy()
        self.emission_forecast = self.target_scaler.inverse_transform(
            self.emission_forecast.reshape(-1, 1)
        ).squeeze()

        return self.emission_forecast


class IntervalPredictor:
    """
    Class that uses CO2Predictors to generate intervals that satisfy conditions 
    (emission in time_step or moving window less than threshold)
    """

    def __init__(self, co2_predictors) -> None:
        self.co2_predictors = co2_predictors

    def predict_intervals(
        self,
        min_step_size=1,
        max_window_size=3,
        max_emission_value=180,
    ):
        """
        This function predicts training intervals.
        Uses co2_predictors to forecast co2 emission, than builds intervals with minimum total emission.
        """
        forecasts = []

        for co2_predictor in self.co2_predictors:
            co2_forecast = co2_predictor.predict_co2()
            forecasts.append(co2_forecast)

        forecasts = np.stack(forecasts)
        time_slots = np.zeros((len(forecasts), 24), dtype=int)

        for forecast_id, forecast in enumerate(forecasts):
            for window_size in range(max_window_size):
                for i in range(len(forecast)):
                    co2_mean = forecast[i : i + min_step_size + window_size].mean()
                    if co2_mean < max_emission_value:
                        time_slots[
                            forecast_id, i : i + min_step_size + window_size - 1
                        ] = 1

        time_slots_vms = np.ones((24), dtype=int) * -1
        for i in range(24):
            if time_slots[:, i].sum() == 0:
                continue
            if time_slots[:, i].sum() == 1:
                time_slots_vms[i] = time_slots[:, i].argmax()
            else:
                min_co2_idx = forecasts[:, i].argmin()
                time_slots_vms[i] = min_co2_idx

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
        else:
            end_time = time
            if current_vm in intervals.keys():
                intervals[current_vm].append((start_time, end_time))
            else:
                intervals[current_vm] = [(start_time, end_time)]

        datetime_intervals = {
            k: [
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
                for jstart, jend in intervals[k]
            ]
            for k in intervals.keys()
        }
        return datetime_intervals


if __name__ == "__main__":
    co2_predictor = CO2Predictor("DNK", 0.5)
    co2_forecast = co2_predictor.predict_co2()
    print(co2_forecast)

    interval_predictor = IntervalPredictor([co2_predictor])
    print(interval_predictor.predict_intervals(max_emission_value=130))

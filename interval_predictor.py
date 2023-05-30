from weather_data_utils import get_points_over_country, get_multiple_last_weather_data
from model import TCNModel
import torch
import numpy as np
import datetime


class IntervalPredictor():
    def __init__(self, country_code, points_step, checkpoint_file_path=None) -> None:
        '''
        Args:
            country_code (str) : ISO-3 coutry code
            points_step (float) : step between points on weather map (in degrees)
            (Optional) checkpoint_file_path (str) : path to file with torch model
        '''
        self.country_points = get_points_over_country(
            country_code, points_step)

        self.lookback_window = 96
        self.predict_window = 24
        self.co2_emission_delta_days = 0

        if checkpoint_file_path is None:
            self.load_model()
        else:
            self.load_model(checkpoint_file_path=checkpoint_file_path)

    def load_model(
        self,
        train=False,
        checkpoint_file_path='default_path.ckpt'
    ):
        """
        This function loads model for predicting time intervals.
        train: bool
            If 'train' is True, then model will be initialized with initial weights and then will be trained on weather and emission data. 
            This is not necessary, this is totaly optional(my fantasy)
        """
        self.predict_model = TCNModel(
            {
                'num_inputs': 22*40,
                'num_channels': [1024, 512, 256, 1],
                'kernel_size': 3,
                'dropout': 0.2
            },
            self.lookback_window, self.predict_window,
            'Adam',
            {'lr': 1e-3, 'weight_decay': 1e-4}
        )
        # self.predict_model = self.predict_model.load_from_checkpoint(
        #     checkpoint_file_path)

    def predict_intervals(
        self,
        time_period,
        min_interval,
        max_emission_value,
    ):
        """
        This function predicts intervals using model for predicting intervals(firstly, it needs to load weather and emission data).
        Returns intervals, saves intervals to the parameter self.intervals

        Args:
            time_period (int) : size of window to calculate average emission 
            min_interval (int) : minimum size of interval to choose
            max_emission_value (float) : decision threshold
        """
        weather_data, weather_time, emission_df = get_multiple_last_weather_data(
            self.country_points, self.lookback_window, co2_emission_delta_days=self.co2_emission_delta_days)

        batch = torch.tensor(weather_data.astype(np.float32))
        batch = batch.unsqueeze(0)

        self.emission_forecast = self.predict_model.predict_step(batch, 0)[0]

        self.emission_forecast = self.emission_forecast.detach().cpu().numpy()

        window_size = time_period

        # takes as input 24h forecasted co2 emission
        # some calculating stuff from Alexey's algorightms
        start = 0
        ii = 0  # time spent for training
        jj = 0  # current time
        flag = True  # flag training is ongoing
        jj_list = []  # list of time, where model was trained
        training_time = 24

        # while current time less that allocated time
        while jj < len(self.emission_forecast):
            min_x = start + jj  # current time

            # Calculate current epoch
            # if the remaining time required to train the model is less than the minimum training step
            if training_time - ii < min_interval:
                step = training_time - ii  # model is trained until target time
                max_x = start + jj + step  # set max time
            else:  # all except the last
                max_x = start + jj + min_interval  # set max time
                step = min_interval  # set step for current session

            # average emission in calculating window
            co2_mean = np.mean(self.emission_forecast[min_x:max_x])

            # calc moving average of emissions in window
            # Пробуем учесть тренд в данных и на восходящем тренде обучать модель.
            # Работает плохо для конца 24-часового отрезка, так как не видим куда пойдет тренд
            predict_window_mean = np.mean(
                self.emission_forecast[min_x:min_x + window_size]) if window_size > 0 else 0

            # condition if average emission in epoch is lower than general average of even moving average
            if co2_mean <= predict_window_mean or co2_mean <= max_emission_value:
                if flag == True:  # start training is ongoing
                    jstart = jj  # get time of training start
                    flag = False  # release flag

                # calculate total emissions for current step emissions
                jj += step  # shift current time
                ii += step  # increase training time
                if ii >= training_time:  # if the time spent on training the model is more than the target
                    break  # stop calculation
            else:  # if condition not met
                if not flag:  # if flag for starting of training not set
                    jend = jj  # get end time of trainining session
                    # save time start and time end of training session
                    jj_list.append((jstart + start, jend + start - 1))
                    flag = True  # release training start flag

                jj += 1  # increase time by one epoch

        if not flag:  # if training stoped
            jend = jj
            # save training time data
            jj_list.append((jstart + start, jend + start - 1))

        datetime_intervals = [(datetime.datetime.now() + datetime.timedelta(hours=jstart),
                               datetime.datetime.now() + datetime.timedelta(hours=jend)) for jstart, jend in jj_list]

        relative_intervals = jj_list
        self.intervals = datetime_intervals
        return datetime_intervals, relative_intervals
    

if __name__ == '__main__':
    predictor = IntervalPredictor('DNK', 0.5)
    datetime_intervals, relative_intervals = predictor.predict_intervals(
        3, 1, 100)

    print(datetime_intervals)
    print(predictor.emission_forecast.shape)

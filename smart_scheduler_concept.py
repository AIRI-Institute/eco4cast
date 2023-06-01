import torch
from torch import nn
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Timer
import tzlocal
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

from interval_predictor import IntervalPredictor
from utils import ResumableRandomSampler, CustomStartProgressBar
from torch.utils.data import DataLoader
from model import ForecastingModel
import datetime
from time import sleep


"""
Что должен делать SmartScheduler:
1) Должен принимать в качестве параметров интервалы времени, модель, датасет, лосс, оптимайзер и тд. У нас будет свой trainer, который будет гонять модель. 
Класс SmartScheduler'а будет обучать модель по батчам на данных ему интервалах нужное число эпох. 
    Плюсы: 
        - Можно будет обучать модель не по эпохам, а по батчам, то есть, одна итерация в цикле обучения - это не эпоха, а батч. 
        Это поможет быть более гибкими в плане интервалов обучения, так как обучение на одном батче обычно не превышает минуты
        - Гибкость настройки. У пользователя будет возможность настроить SmartScheduler под себя
    Минусы: 
        - Сложно в плане разработки. Придется ввести поддержку callback'ов, чтобы пользователи могли кастомизировать trainer.
        Придется ввести очень много условий в trainer, чтобы его функционал мог удовлетворить нуждам большинства пользователей
        - Сложно в плане пользования. При необходимости модифицировать trainer придется потратить немалое время.
        Это может быть проблематичным для некоторых пользователей. 

2) Должен принимать на вход функцию и ее аргументы. SmartScheduler будет запускать функцию столько раз, сколько потребуется, или пока не закончатся интервалы времени
    Плюсы: 
        - Относительно легко в реализции. Нам просто надо реализовать executor, который будет запускать данную ему функцию в данные ему интервалы времени. 
        - Простота в использовании
    Минусы: 
        - Отсутсвие гибкости. В случае машинного обучения подаваемой функцией будет эпоха(почти наверно), а эпоха иногда очень долго обучается. 
        Это значит, что накладывается соответствующее ограничение на длину интервала для обучения
"""


class IntervalTrainer:
    """
    This class is help class for SmartSchedulerTrainer.
    It trains model during datetime intervals using BackgroundScheduler
    """

    # TODO : some usage of test dataset or delete it?
    # __free_memory method
    #

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        loss_function: torch.nn.Module,
        metric_func,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        lr=1e-4,
        epochs=3,
        val_step=None,
        batch_size=8,
        device="cpu",
        show_val_progressbar=True,
    ):
        """
        val_step: int
            Every val_step training steps one will make validation
            if None - validates model after training epoch
        metric_func: func
            This func returns float number to choose best model after validation
        Other parameters are obvious and I won't describe it
        """

        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.loss_function = loss_function
        self.epochs = epochs
        self.metric_func = metric_func
        self.val_step = val_step if val_step is not None else 1e9

        self.train_sampler = ResumableRandomSampler(self.train_dataset)
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=batch_size, sampler=self.train_sampler
        )

        self.val_sampler = ResumableRandomSampler(self.val_dataset)
        self.val_dataloader = DataLoader(
            self.val_dataset, batch_size=batch_size, sampler=self.val_sampler
        )
        self.training_state = "train"
        self.last_train_batch_idx = 0
        self.last_val_batch_idx = 0
        self.last_epoch = 0
        self.has_states_to_load = False

        self.__scheduler = BackgroundScheduler(
            misfire_grace_time=3600,
            job_defaults={"misfire_grace_time": 3600},
        )
        self.device = device
        self.val_loss = 0
        self.train_loss = 0
        self.train_metric = 0
        self.val_metric = 0
        self.best_val_metric = 1e9
        self.show_val_progressbar = show_val_progressbar

    def __load_states(self):
        if self.has_states_to_load:
            self.train_sampler.set_state(torch.load("last_train_sampler_state.pth"))
            self.val_sampler.set_state(torch.load("last_val_sampler_state.pth"))
            self.model.load_state_dict(torch.load("last_model_state.pth"))

    def __save_states(self):
        self.has_states_to_load = True
        torch.save(self.train_sampler.get_state(), "last_train_sampler_state.pth")
        torch.save(self.val_sampler.get_state(), "last_val_sampler_state.pth")
        torch.save(self.model.state_dict(), "last_model_state.pth")
        # Maybe save loss, metrics and other self.* params?

    def __free_memory(self):
        # Remove everything from CUDA, but keep model architecture on CPU for future loading
        pass

    def __train(self, end_time):
        self.model.train()
        train_progress_bar = CustomStartProgressBar(
            len(self.train_dataloader), self.last_train_batch_idx, description="Train"
        )
        if self.training_state == "train_val":
            self.__validate(end_time)

        for batch_id, batch in enumerate(self.train_dataloader):
            if datetime.datetime.now(datetime.timezone.utc) >= end_time:
                self.shutting = True

            if self.shutting:
                self.last_train_batch_idx += batch_id
                break

            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model.forward(x)
            loss = self.loss_function(logits, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.train_loss += loss.item() / len(self.train_dataloader)
            self.train_metric += self.metric_func(y, logits) / len(
                self.train_dataloader
            )

            train_progress_bar()

            if (
                self.last_train_batch_idx + batch_id > 0
                and (self.last_train_batch_idx + batch_id) % self.val_step == 0
            ):
                self.training_state = "train_val"
                self.__validate(end_time)

        else:
            self.last_train_batch_idx = 0
            self.training_state = "val"

    def __validate(self, end_time):
        self.model.eval()

        val_progress_bar = (
            CustomStartProgressBar(
                len(self.val_dataloader), self.last_val_batch_idx, description="Val"
            )
            if self.show_val_progressbar
            else lambda: None
        )

        with torch.no_grad():
            for batch_id, batch in enumerate(self.val_dataloader):
                if self.shutting:
                    break

                if datetime.datetime.now(datetime.timezone.utc) >= end_time:
                    self.shutting = True
                    self.last_val_batch_idx += batch_id
                    break

                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model.forward(x)
                loss = self.loss_function(logits, y)
                self.val_loss += loss.item() / len(self.val_dataloader)
                self.val_metric += self.metric_func(y, logits) / len(
                    self.val_dataloader
                )

                val_progress_bar()
                self.last_val_batch_idx = batch_id

            else:
                # print(f'Val {batch_id} steps ended')
                if self.val_metric < self.best_val_metric:
                    self.best_val_metric = self.val_metric
                    torch.save(self.model.state_dict(), "best_model.pth")
                    print(
                        "\n",
                        f"Saving new best model with metric {self.best_val_metric}",
                    )

                self.val_metric = 0
                self.last_val_batch_idx = 0
                self.training_state = "train"

    def __train_loop(self, end_time):
        self.__load_states()

        print(
            f"Starting train from epoch {self.last_epoch}, state: {self.training_state},"
            + f"train_batch {self.last_train_batch_idx}, val_batch {self.last_val_batch_idx}"
        )
        self.shutting = False
        for self.last_epoch in range(self.last_epoch, self.epochs):
            if (
                self.training_state == "train"
                or self.training_state == "train_val"
                and not self.shutting
            ):
                self.__train(end_time)

            if self.training_state == "val" and not self.shutting:
                self.__validate(end_time)

            if self.shutting:
                self.__save_states()
                self.__free_memory()
                self.shutting = False
                print("\n", "Shutting training till next interval")
                break

            print(
                f"Epoch {self.last_epoch} \t Training Loss: {self.train_loss} \t"
                + f"Validation Loss: {self.val_loss}"
            )

            self.train_loss = 0
            self.val_loss = 0
            self.train_metric = 0

        else:
            self.last_epoch += 1

    def train(
        self,
        datetime_intervals=None,
    ):
        """
        This function will train model for 'epochs' epochs or until total time in time intervals will pass
        If intervals == None, training process runs as usual.
        """
        if datetime_intervals is None:
            self.__train_loop(
                datetime.datetime(
                    year=2500, month=1, day=1, tzinfo=datetime.timezone.utc
                )
            )
        else:
            self.__scheduler.start()
            for start_interval, end_interval in datetime_intervals:
                if self.last_epoch >= self.epochs:
                    break

                print(f"Scheduling {start_interval} - {end_interval} job")
                trigger = DateTrigger(run_date=start_interval)
                self.__scheduler.add_job(
                    func=self.__train_loop,
                    trigger=trigger,
                    args=[end_interval],
                    id=f"job",
                )

                # 5 seconds for saving states
                waiting_till = end_interval + datetime.timedelta(seconds=5)
                try:
                    while (
                        datetime.datetime.now(datetime.timezone.utc) < waiting_till
                        and self.last_epoch < self.epochs
                    ):
                        sleep(1)
                except (KeyboardInterrupt, SystemExit):
                    print("\n", "KeyboardInterrupt caught. Stopping scheduled jobs")
                    self.__scheduler.remove_all_jobs()
                    self.__scheduler.shutdown(wait=False)
                    self.shutting = True
                    break


class SmartSchedulerTrainer:
    """
    We can implement this class so that it trains a trainer with BackgroundScheduler(https://apscheduler.readthedocs.io/en/3.x/modules/schedulers/background.html)
    or normally, like usual training process.
    """

    def __init__(
        self,
        trainer: IntervalTrainer,
        interval_predictor: IntervalPredictor,
    ):
        self.trainer = trainer
        self.interval_predictor = interval_predictor
        pass

    def train(self, time_period=3, min_interval=1, max_emission_value=100.0):
        """
        This func to start trainer on forecasted intervals by interval_predictor
        Args:
            time_period (int) : lenght of emission data history to use in interval prediction. And size of moving window
            min_interval (int) : minimal training interval in hours
            max_emission_value (float) : decision threshold for co2 emission. For example, historic mean value
        """
        self.intervals = self.interval_predictor.predict_intervals(
            time_period=time_period,
            min_interval=min_interval,
            max_emission_value=max_emission_value,
        )
        self.trainer.train(self.intervals)


class SmartSchedulerFunction:
    """ """

    def __init__(self, function, interval_predictor: IntervalPredictor):
        self.function = function
        self.interval_predictor = interval_predictor
        self.__scheduler = BackgroundScheduler(
            misfire_grace_time=None,
            job_defaults={"misfire_grace_time": None},
        )

    def predict_intervals(
        self, time_period=3, min_interval=1, max_emission_value=100.0
    ):
        """
        This function predicts intervals using model for predicting intervals(firstly, it needs to load weather and emission data).
        Returns intervals, saves intervals to the parameter self.intervals

        Args:
            time_period (int) : lenght of emission data history to use in interval prediction. And size of moving window
            min_interval (int) : minimal training interval in hours
            max_emission_value (float) : decision threshold for co2 emission. For example, historic mean value
        """

        datetime_intervals = self.interval_predictor.predict_intervals(
            time_period, min_interval, max_emission_value
        )
        self.intervals = datetime_intervals
        return datetime_intervals

    def start(self):
        """
        This function may call either start or schedule. No matter.
        It starts function running.
        """

        self.__scheduler.start()
        for start_interval, end_interval in self.intervals:
            print(f"Scheduling {start_interval} - {end_interval} job")
            trigger = DateTrigger(run_date=start_interval)
            self.__scheduler.add_job(
                func=self.__train_loop,
                trigger=trigger,
                args=[end_interval],
                id=f"job",
            )

            waiting_till = end_interval + datetime.timedelta(seconds=5)
            try:
                while datetime.datetime.now(datetime.timezone.utc) < waiting_till:
                    sleep(1)
            except (KeyboardInterrupt, SystemExit):
                print("\n", "KeyboardInterrupt caught. Stopping scheduled jobs")
                self.__scheduler.remove_all_jobs()
                self.__scheduler.shutdown(wait=False)
                break

    def stop(self):
        """
        This method stops running a function.
        """
        self.__scheduler.remove_all_jobs()
        self.__scheduler.shutdown(wait=False)

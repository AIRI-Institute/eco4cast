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
    I think it should be similar to this class: https://huggingface.co/docs/transformers/main_classes/trainer
    It's purpose to take training parameters and train model
    """

    # TODO: some class and methohds descriptions
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
        epochs=100,
        val_step=None,
        batch_size=8,
        device="cpu",
        show_val_progressbar=True,
    ):
        """
        val_step: int
            Every val_step training steps one will make validation
        metric_func: func
            This func returns either dict with metrics or just float number
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

        self.__scheduler = BackgroundScheduler(misfire_grace_time=None)
        self.device = device
        self.val_loss = 0
        self.train_loss = 0
        self.train_metric = 0
        self.val_metric = 0
        self.best_val_metric = 1e10
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
        # TODO : metric collection and
        if self.training_state == "train_val":
            self.__validate(end_time)

        for batch_id, batch in enumerate(self.train_dataloader):
            if datetime.datetime.now(datetime.timezone.utc) >= end_time:
                self.shutting = True
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

            if batch_id > 0 and batch_id % self.val_step == 0:
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
                    print(f"Saving new best model with metric {self.best_val_metric}")

                self.val_metric = 0
                self.last_val_batch_idx = 0
                self.training_state = "train"

    def train_loop(self, end_time):
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
                print("Shutting training till next interval")
                break

            print(
                f"Epoch {self.last_epoch} \t Training Loss: {self.train_loss} \t"
                + f"Validation Loss: {self.val_loss}"
            )

            self.train_loss = 0
            self.val_loss = 0
            self.train_metric = 0

    def train(
        self,
        datetime_intervals=None,
    ):
        """
        This function will train model for 'epochs' epochs or until total time in time intervals will pass
        If intervals == None, training process runs as usual.
        """
        if datetime_intervals is None:
            self.train_loop(
                datetime.datetime(
                    year=2500, month=1, day=1, tzinfo=datetime.timezone.utc
                )
            )
        else:
            self.__scheduler.start()
            for start_interval, end_interval in datetime_intervals:
                # This logic schedules model training for all intervals at once.
                # Maybe we should wait till the previous interval ends or is it okay?

                print(f"Scheduling {start_interval} - {end_interval} job")
                trigger = DateTrigger(run_date=start_interval)

                # trigger = CronTrigger(start_date=start_interval, end_date=end_interval)
                self.__scheduler.add_job(
                    func=self.train_loop,
                    trigger=trigger,
                    args=[end_interval],
                    id=f"job",
                )

                waiting_till = end_interval + datetime.timedelta(seconds=10)
                while datetime.datetime.now(datetime.timezone.utc) < waiting_till:
                    sleep(1)

                # self.__scheduler.remove_job(f"job")
                # self.__scheduler.shutdown()


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

    def train(self):
        """
        This func trains self.trainer
        """
        self.intervals = self.interval_predictor.predict_intervals(3, 1, 100)
        self.trainer.train(self.intervals)
        pass


class SmartSchedulerFunction:
    """ """

    def __init__(self, function, interval_predictor):
        self.function = function
        self.interval_predictor = interval_predictor
        self.__scheduler = BackgroundScheduler(
            timezone=str(tzlocal.get_localzone()), misfire_grace_time=None
        )

    def predict_intervals(
        self,
        time_period,
        min_interval,
        max_emission_value,
    ):
        """
        This function predicts intervals using model for predicting intervals(firstly, it needs to load weather and emission data).
        Returns intervals, saves intervals to the parameter self.intervals
        """
        (
            datetime_intervals,
            relative_intervals,
        ) = self.interval_predictor.predict_intervals(
            time_period, min_interval, max_emission_value
        )
        self.intervals = datetime_intervals
        return datetime_intervals

    def start(self):
        """
        This function may call either start or schedule. No matter.
        It starts function running.
        """

        first_interval = self.intervals[0]

        self.__scheduler.start()
        trigger = DateTrigger(run_date=first_interval[0])
        # print(start_date, end_date)
        self.__scheduler.add_job(
            func=self.function,
            trigger=trigger,
            args=self.__args,
            kwargs=self.__kwargs,
            id="job",
        )

    def stop(self):
        """
        This method stops running a function.
        """
        pass

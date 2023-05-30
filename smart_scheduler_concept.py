import torch
from torch import nn
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Timer
import tzlocal
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

from interval_predictor import IntervalPredictor

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


class IntervalTrainer():
    """
    This class is help class for SmartSchedulerTrainer. 
    I think it should be similar to this class: https://huggingface.co/docs/transformers/main_classes/trainer
    It's purpose to take training parameters and train model
    """

    def __init__(
            self,
            model,
            train_loader,
            val_loader,
            test_loader,
            optimizer,
            loss_function,
            epochs,
            metric_func,
            val_step,
    ):
        """
        val_step: int
            Every val_step training steps one will make validation
        metric_func: func
            This func returns either dict with metrics or just float number  
        Other parameters are obvious and I won't describe it
        """
        # self.lightning_model = lightning_model
        # self.data_module = data_module
        # self.epochs = epochs
        
        pass

    def train(
            self,
            datetime_intervals=None,
    ):
        """
        This function will train model for 'epochs' epochs or until total time in time intervals will pass
        If intervals == None, training process runs as usual.
        """
        for start_interval, end_interval in datetime_intervals:
            duration = end_interval-start_interval
            timer = Timer(duration=duration, interval='step', verbose=True)
            trainer = Trainer(callbacks=[timer])

        pass


class SmartSchedulerTrainer():
    """
    We can implement this class so that it trains a trainer with BackgroundScheduler(https://apscheduler.readthedocs.io/en/3.x/modules/schedulers/background.html)
    or normally, like usual training process.
    """

    def __init__(
        self,
        trainer : IntervalTrainer,
        interval_predictor : IntervalPredictor,
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


class SmartSchedulerFunction():
    """

    """

    def __init__(
        self,
        function,
        interval_predictor
    ):
        self.function = function
        self.interval_predictor = interval_predictor
        self.__scheduler = BackgroundScheduler(
            timezone=str(tzlocal.get_localzone()),
            misfire_grace_time=None
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
        datetime_intervals, relative_intervals = self.interval_predictor.predict_intervals(
            time_period, min_interval, max_emission_value)
        self.intervals = datetime_intervals
        return datetime_intervals

    def start(self):
        """
        This function may call either start or schedule. No matter. 
        It starts function running. 
        """

        first_interval = self.intervals[0]

        self.__scheduler.start()
        trigger = DateTrigger(
                run_date=first_interval[0]
            )
        # print(start_date, end_date)
        self.__scheduler.add_job(
            func=self.function, 
            trigger=trigger, 
            args=self.__args, 
            kwargs=self.__kwargs,
            id="job")

    def stop(self):
        """
        This method stops running a function. 
        """
        pass

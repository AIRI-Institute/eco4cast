from ast import List
import torch
from torch import nn
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger
from smartscheduler.virtual_machine.utils import (
    ResumableRandomSampler,
    CustomStartProgressBar,
)
from torch.utils.data import DataLoader
import datetime
from time import sleep
from copy import deepcopy
import gc
from lightning.fabric import Fabric
from lightning.fabric.utilities.types import _Stateful
import eco2ai
import os


class IntervalTrainer:
    """
    This class is help class for SmartSchedulerTrainer.
    It trains model during datetime intervals using BackgroundScheduler
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
        loss_function: torch.nn.Module,
        metric_func,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        lr=1e-4,
        epochs=3,
        val_step=None,
        batch_size=8,
        device="cpu",
        show_progressbar=True,
        callbacks: List = None,
        project_name="noname_project",
        country_code_alpha_2=None,
        num_workers=1,
    ):
        """
        val_step: int
            Every val_step training steps one will make validation
            if None - validates model after training epoch
        metric_func: func
            This func returns float number to choose best model after validation
        Other parameters are obvious
        """
        self.optimizer_class = optimizer
        self.lr = lr
        self.model_arch = model.to("cpu")
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.loss_function = loss_function
        self.epochs = epochs
        self.metric_func = metric_func
        self.val_step = val_step if val_step is not None else 1e9

        self.train_sampler = ResumableRandomSampler(self.train_dataset)
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            sampler=self.train_sampler,
            num_workers=num_workers,
            persistent_workers=True
        )

        self.val_sampler = ResumableRandomSampler(self.val_dataset)
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            sampler=self.val_sampler,
            num_workers=num_workers,
            persistent_workers=True
        )
        self.training_state = "train"
        self.last_train_batch_idx = 0
        self.last_val_batch_idx = 0
        self.last_epoch = 0
        self.has_states_to_load = False

        self.__scheduler = BackgroundScheduler(
            misfire_grace_time=7200,
            job_defaults={"misfire_grace_time": 7200},
        )
        self.device = device
        self.val_loss = 0
        self.train_loss = 0
        self.train_metric = 0
        self.val_metric = 0
        self.best_val_metric = 1e9
        self.show_progressbar = show_progressbar

        self.fabric = Fabric(accelerator=self.device, callbacks=callbacks)
        self.train_dataloader, self.val_dataloader = self.fabric.setup_dataloaders(
            self.train_dataloader, self.val_dataloader
        )

        self.project_name = project_name
        self.country_code_alpha_2 = country_code_alpha_2
        self.callbacks = callbacks

    def __update_self_state(self):
        self.state = {
            "train_sampler": self.train_sampler,
            "val_sampler": self.val_sampler,
            "model": self.model,
            "optimizer": self.optimizer,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "best_val_metric": self.best_val_metric,
            "training_state": self.training_state,
            "last_train_batch_idx": self.last_train_batch_idx,
            "last_val_batch_idx": self.last_val_batch_idx,
            "last_epoch": self.last_epoch,
            "project_name": self.project_name,
            "callbacks": self.callbacks,
        }

    def __load_states(self):
        self.model = deepcopy(self.model_arch)
        self.optimizer: torch.optim.Optimizer = self.optimizer_class(
            self.model.parameters(), lr=self.lr
        )

        self.model, self.optimizer = self.fabric.setup(
            self.model, self.optimizer_class(self.model.parameters(), lr=self.lr)
        )

        self.__update_self_state()

        if self.has_states_to_load:
            checkpoint = self.fabric.load("fabric_checkpoint.ckpt")
            for name, obj in self.state.copy().items():
                if isinstance(obj, _Stateful) or isinstance(obj, nn.Module):
                    obj.load_state_dict(checkpoint.pop(name))
            self.train_loss = checkpoint["train_loss"]
            self.val_loss = checkpoint["val_loss"]
            self.best_val_metric = checkpoint["best_val_metric"]
            self.training_state = checkpoint["training_state"]
            self.last_train_batch_idx = checkpoint["last_train_batch_idx"]
            self.last_val_batch_idx = checkpoint["last_val_batch_idx"]
            self.last_epoch = checkpoint["last_epoch"]
            self.project_name = checkpoint["project_name"]
            self.callbacks = checkpoint["callbacks"]

    def __save_states(self):
        self.has_states_to_load = True
        self.__update_self_state()
        self.fabric.save("fabric_checkpoint.ckpt", self.state)

    def __free_memory(self):
        # Remove everything from CUDA, but keep model architecture on CPU for future loading
        # Honestly, it does not clear CUDA completely, around 500 MB are kept on device
        del self.model
        del self.optimizer
        gc.collect()
        torch.cuda.empty_cache()

    def __train_epoch(self, end_time):
        self.model.train()
        train_progress_bar = (
            CustomStartProgressBar(
                len(self.train_dataloader),
                self.last_train_batch_idx,
                description="Train",
            )
            if self.show_progressbar
            else lambda: None
        )

        if self.training_state == "train_val":
            self.__val_epoch(end_time)

        if self.last_val_batch_idx == 0:
            self.fabric.call("on_train_epoch_start", self)

        for batch_id, batch in enumerate(self.train_dataloader):
            if datetime.datetime.now(datetime.timezone.utc) >= end_time:
                self.shutting = True

            if self.shutting:
                self.last_train_batch_idx += batch_id
                break

            self.fabric.call("on_train_batch_start", self, batch, batch_id)
            x, y = batch
            logits = self.model.forward(x)
            loss = self.loss_function(logits, y)
            self.fabric.call("on_before_backward", self, loss)
            self.fabric.backward(loss)
            self.fabric.call("on_after_backward")
            self.fabric.call("on_before_optimizer_step", self, self.optimizer)
            self.optimizer.step()
            self.fabric.call("on_before_zero_grad", self, self.optimizer)
            self.optimizer.zero_grad()

            self.train_loss += loss.item() / len(self.train_dataloader)
            self.train_metric += self.metric_func(y, logits) / len(
                self.train_dataloader
            )

            train_progress_bar()

            self.fabric.call("on_train_batch_end", self, loss=loss, output=logits)

            if (
                self.last_train_batch_idx + batch_id > 0
                and (self.last_train_batch_idx + batch_id) % self.val_step == 0
            ):
                self.training_state = "train_val"
                self.__val_epoch(end_time)

        else:
            self.last_train_batch_idx = 0
            self.training_state = "val"
            self.fabric.call("on_train_epoch_end", self)

    def __val_epoch(self, end_time):
        self.model.eval()

        val_progress_bar = (
            CustomStartProgressBar(
                len(self.val_dataloader), self.last_val_batch_idx, description="Val"
            )
            if self.show_progressbar
            else lambda: None
        )
        self.fabric.call("on_validation_epoch_start", self)
        with torch.no_grad():
            for batch_id, batch in enumerate(self.val_dataloader):
                if self.shutting:
                    break

                if datetime.datetime.now(datetime.timezone.utc) >= end_time:
                    self.shutting = True
                    self.last_val_batch_idx += batch_id
                    break

                self.fabric.call("on_validation_batch_start", self, batch, batch_id)
                x, y = batch
                logits = self.model.forward(x)
                loss = self.loss_function(logits, y)
                self.val_loss += loss.item() / len(self.val_dataloader)
                self.val_metric += self.metric_func(y, logits) / len(
                    self.val_dataloader
                )

                val_progress_bar()
                self.last_val_batch_idx = batch_id
                self.fabric.call(
                    "on_validation_batch_end", self, logits, batch, batch_id
                )

            else:
                self.fabric.call("on_validation_epoch_end", self, self.val_metric)
                self.val_metric = 0
                self.last_val_batch_idx = 0
                self.training_state = "train"

    def __train_loop(self, end_time):
        self.emission_tracker.start()
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
                self.__train_epoch(end_time)

            if self.training_state == "val" and not self.shutting:
                self.__val_epoch(end_time)

            if self.shutting:
                self.__save_states()
                self.__free_memory()
                self.shutting = False
                self.emission_tracker.stop()
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
            self.emission_tracker.stop()
            self.last_epoch = self.epochs + 1

    def __init_emission_tracker(self, description=''):
        if not os.path.exists(f"{self.project_name}_emissions"):
            os.mkdir(f"{self.project_name}_emissions")

        self.emission_tracker = eco2ai.Tracker(
            project_name=self.project_name,
            experiment_description=description,
            file_name=f"{self.project_name}_emissions/{self.project_name}_{description}.csv",
            alpha_2_code=self.country_code_alpha_2,
        )

    def stop_training(
        self,
    ):
        self.shutting = True
        self.last_epoch = self.epochs
        print("End of training.")

    def train(self, datetime_intervals=None, load_states=False):
        """
        This function will train model for 'epochs' epochs or until total time in time intervals will pass
        If intervals == None, training process runs as usual.
        """
        self.has_states_to_load = load_states
        if datetime_intervals is None:
            self.__init_emission_tracker()
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

                self.__init_emission_tracker()
                print(f"Scheduling {start_interval} - {end_interval} job")
                trigger = DateTrigger(run_date=start_interval)
                self.__scheduler.add_job(
                    func=self.__train_loop,
                    trigger=trigger,
                    args=[end_interval],
                    id=f"job",
                )

                # self.__train_loop(end_interval)

                # 30 extra seconds for saving states
                waiting_till = end_interval + datetime.timedelta(seconds=30)
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

                del self.emission_tracker

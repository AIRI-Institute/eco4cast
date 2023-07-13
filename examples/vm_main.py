from smartscheduler.virtual_machine.smart_scheduler_concept import IntervalTrainer
from torch import nn
import torch.nn.functional as F
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
import argparse
import datetime


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
train_dataset = datasets.MNIST(
    "mnist_data", train=True, download=True, transform=transform
)
train_dataset = Subset(train_dataset, list(range(10000)))
val_dataset = datasets.MNIST("mnist_data", train=False, transform=transform)
val_dataset = Subset(val_dataset, list(range(5000)))


def accuracy(labels: torch.tensor, logits: torch.tensor):
    preds = logits.argmax(1)
    return (preds == labels).float().mean()


class EarlyStoppingCallback:
    def __init__(self, patience=5, mode="max") -> None:
        self.patience = patience
        self.metric_history = []
        assert mode in ["min", "max"]

        self.best_val_metric = 1e9 if mode == "min" else -1e9
        self.mode = mode

    def on_validation_epoch_end(
        self,
        trainer: IntervalTrainer,
        val_metric,
    ):
        val_metric = float(val_metric)
        self.metric_history.append(val_metric)
        if self.mode == "min":
            self.best_val_metric = min(self.best_val_metric, val_metric)
        else:
            self.best_val_metric = max(self.best_val_metric, val_metric)

        if self.best_val_metric not in self.metric_history[-self.patience - 1 :]:
            print("\n", "EarlyStopping")
            trainer.stop_training()


class BestModelSavingCallback:
    def __init__(self, mode="max") -> None:
        assert mode in ["min", "max"]
        self.best_val_metric = 1e9 if mode == "min" else -1e9
        self.mode = mode

    def on_validation_epoch_end(
        self,
        trainer: IntervalTrainer,
        val_metric,
    ):
        if (self.mode == "min" and val_metric < self.best_val_metric) or (
            self.mode == "max" and val_metric > self.best_val_metric
        ):
            self.best_val_metric = val_metric
            torch.save(trainer.model.state_dict(), "best_model.pth")
            print(
                "\n",
                f"Saving new best model with metric {self.best_val_metric}",
            )


parser = argparse.ArgumentParser()
parser.add_argument("--times", type=str)
parser.add_argument("--load_states", action="store_true")
args = parser.parse_args()

intervals = [
    datetime.datetime.strptime(item + "+00:00", "%Y%m%d%H%M%S%z")
    for item in args.times.split(",")
]
intervals = [(intervals[i], intervals[i + 1]) for i in range(0, len(intervals), 2)]
load_states = args.load_states


# callbacks = [EarlyStoppingCallback(100), BestModelSavingCallback()]
callbacks = [BestModelSavingCallback()]


trainer = IntervalTrainer(
    model=Net(),
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    loss_function=nn.CrossEntropyLoss(),
    metric_func=accuracy,
    val_step=None,
    show_val_progressbar=True,
    epochs=20,
    device="cpu",
    callbacks=callbacks,
    project_name="MNIST_example",
)


trainer.train(intervals, load_states)

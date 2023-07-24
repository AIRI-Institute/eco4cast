from smartscheduler.virtual_machine.smart_scheduler_concept import IntervalTrainer
from torch import nn
import torch
from torchvision import datasets, transforms
from torchvision.models import resnet101, ResNet101_Weights
import argparse
import datetime


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



transform = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = datasets.ImageFolder(root='ImageNet_dataset/ILSVRC/Data/CLS-LOC/train',transform = transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

def accuracy(labels: torch.tensor, logits: torch.tensor):
    preds = logits.argmax(1)
    return (preds == labels).float().mean()


# load_states = False
# intervals = [(
#     datetime.datetime(2023, 7, 24, 10, 00, tzinfo=datetime.timezone.utc),
#     datetime.datetime(2023, 7, 24, 10, 30, tzinfo=datetime.timezone.utc),
# ),
# (
#     datetime.datetime(2023, 7, 24, 10, 31, tzinfo=datetime.timezone.utc),
#     datetime.datetime(2023, 7, 30, 5, 10, tzinfo=datetime.timezone.utc),
# )]


callbacks = [EarlyStoppingCallback(10), BestModelSavingCallback()]
# callbacks = [BestModelSavingCallback()]


trainer = IntervalTrainer(
    model=resnet101(),
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    loss_function=nn.CrossEntropyLoss(),
    metric_func=accuracy,
    val_step=None,
    show_progressbar=True,
    epochs=3,
    device="gpu",
    callbacks=callbacks,
    project_name="ImageNet_example",
    batch_size=32,
    num_workers=6,
)


trainer.train(intervals, load_states)

from smart_scheduler_concept import IntervalTrainer
from torch import nn
import torch.nn.functional as F
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
import datetime

# MNIST example


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
    "../data", train=True, download=True, transform=transform
)
train_dataset = Subset(train_dataset, list(range(10000)))
val_dataset = datasets.MNIST("../data", train=False, transform=transform)
val_dataset = Subset(val_dataset, list(range(5000)))

loss = nn.CrossEntropyLoss()


def metric_fun(labels: torch.tensor, logits: torch.tensor):
    preds = logits.argmax(1)
    return (preds == labels).float().mean()


trainer = IntervalTrainer(
    model=Net(),
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=None,
    loss_function=loss,
    metric_func=metric_fun,
    val_step=700,
    show_val_progressbar=True,
    epochs=2,
)

intervals = [
    (
        datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=0),
        datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=26),
    ),
    (
        datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=35),
        datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=160),
    ),
]


# Can be used both ways
trainer.train(intervals)
# trainer.train(None)

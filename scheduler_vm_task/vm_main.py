from smart_scheduler_concept import IntervalTrainer
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


class MyCallback:
    def on_train_batch_end(self, loss, output):
        # print(2 * loss)
        pass


trainer = IntervalTrainer(
    model=Net(),
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=None,
    loss_function=nn.CrossEntropyLoss(),
    metric_func=accuracy,
    val_step=None,
    show_val_progressbar=True,
    epochs=20,
    device="cpu",
    callbacks=[MyCallback()],
    project_name="MNIST_example",
)


parser = argparse.ArgumentParser()
parser.add_argument("--times", type=str)
parser.add_argument("--load_states", action="store_true")
args = parser.parse_args()

intervals = [
    datetime.datetime.strptime(item+'+00:00', "%Y%m%d%H%M%S%z") for item in args.times.split(",")
]
intervals = [(intervals[i], intervals[i + 1]) for i in range(0, len(intervals), 2)]
load_states = args.load_states

trainer.train(intervals, load_states)

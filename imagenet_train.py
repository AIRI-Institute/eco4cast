import eco2ai
from lightning import LightningModule
from torchvision import datasets
from torchvision.models import resnet101
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from lightning.pytorch import Trainer
import os



emission_tracker = eco2ai.Tracker('ImageNet_emission', file_name='imagenet_emission.csv')
emission_tracker.start()

transform = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


path = 'ImageNet_dataset/ILSVRC/Data/CLS-LOC/train'

dataset = datasets.ImageFolder(root=path,transform = transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(
    train_set,
    batch_size=32,
    num_workers=6,
    shuffle=True
)

val_loader = DataLoader(
    val_set,
    batch_size=32,
    num_workers=6,
    shuffle=False
)



class Resnet(LightningModule):
    def __init__(
        self,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.loss_module = nn.CrossEntropyLoss()
        self.resnet = resnet101()


    def forward(self, x: torch.Tensor):
        out = self.resnet(x)
        return out


    def predict_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        return preds

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss_module(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss_module(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-5)    
        return [optimizer]
    

trainer = Trainer(
    accelerator="gpu",
    devices=[0],
    max_epochs=2,
    callbacks=[],
    default_root_dir="imagenet_models",
)

trainer.fit(Resnet(), train_loader, val_loader)

emission_tracker.stop()
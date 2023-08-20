import gc
import os
import pathlib
import pprint
import urllib
from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchmetrics.functional.classification.accuracy import accuracy
from torchvision import datasets, transforms

# wczytanie danych

class_names = [
    "airplane",
    "banana",
    "cookie",
    "diamond",
    "dog",
    "hot air balloon",
    "knife",
    "parachute",
    "scissors",
    "wine glass",
]
data_folder = "../data/quickdraw/"

# make sure data_folder exists - pathlib
pathlib.Path(data_folder).mkdir(parents=True, exist_ok=True)


class QuickDrawDataset(Dataset):
    """A Quick, Draw! dataset"""

    def __init__(
        self, classes, root_dir, download_data=False, load_data=True, transform=None
    ):
        """
        Arguments:
            classes (list[string]): List of classes to be used.
            root_dir (string): Directory with all the images.
            download (bool, optional) – If True, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.
        """
        self.classes = classes
        self.root_dir = root_dir

        if download_data:
            self.download_data()

        if load_data:
            self.data, self.targets = self._load_data()

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img, target = self.data[idx], int(self.targets[idx])

        if self.transform:
            img = self.transform(img)

        return img, target

    def download_data(self):
        for name in self.classes:
            url = (
                "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/%s.npy"
                % name
            )
            file_name = self.root_dir + url.split("/")[-1].split("?")[0]

            url = url.replace(" ", "%20")

            if not os.path.isfile(file_name):
                print(url, "==>", file_name)
                urllib.request.urlretrieve(url, file_name)

    def _load_data(self):
        raw_data = []
        for name in self.classes:
            file_name = self.root_dir + name + ".npy"
            raw_data.append(np.load(file_name, fix_imports=True, allow_pickle=True))
            print("%-15s" % name, type(raw_data[-1]))

        reshaped_data = np.concatenate(raw_data).reshape(-1, 28, 28, 1)
        reshaped_targets = np.concatenate(
            [np.full(d.shape[0], i) for i, d in enumerate(raw_data)]
        )

        return reshaped_data, reshaped_targets

    def _set_data(self, data, targets):
        self.data = data
        self.targets = targets

    def split_train_test(self, test_size=0.2):
        """Split data into train and test sets using sklearn.model_selectiontrain_test_split function."""

        X_train, X_test, y_train, y_test = train_test_split(
            self.data,
            self.targets,
            test_size=test_size,
            random_state=12,
            stratify=self.targets,
        )

        train_dataset = QuickDrawDataset(
            self.classes,
            self.root_dir,
            download_data=False,
            load_data=False,
            transform=self.transform,
        )
        test_dataset = QuickDrawDataset(
            self.classes,
            self.root_dir,
            download_data=False,
            load_data=False,
            transform=self.transform,
        )

        train_dataset._set_data(X_train, y_train)
        test_dataset._set_data(X_test, y_test)

        return train_dataset, test_dataset


all_dataset = QuickDrawDataset(
    class_names,
    data_folder,
    download_data=True,
    load_data=True,
    transform=transforms.ToTensor(),
)

train_dataset, test_dataset = all_dataset.split_train_test(test_size=0.2)

# to save RAM
del all_dataset
gc.collect()

print(f"train_dataset: {len(train_dataset)} samples")
print(f"test_dataset: {len(train_dataset)} samples")


def get_torch_optimizer(optimizer_name, model_params, lr):
    if optimizer_name == "Adam":
        return torch.optim.Adam(
            model_params,
            lr=lr,
        )
    elif optimizer_name == "SGD":
        return torch.optim.SGD(
            model_params,
            lr=lr,
        )
    else:
        raise ValueError(f"Unknown optimizer {optimizer_name}")


def get_torch_loss(loss_name):
    if loss_name == "cross_entropy":
        return torch.nn.CrossEntropyLoss()
    elif loss_name == "mse":
        return torch.nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss {loss_name}")


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


config = {
    "batch_size": 64,
    "epochs": 10,
    "learning_rate": 1e-3,
    "loss": "cross_entropy",
    "optimizer": "Adam",
    "device": ("cuda" if torch.cuda.is_available() else "cpu"),
}

train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"])
test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"])


class QuickDrawNetwork_V1(nn.Module):
    def __init__(self, dimensions, num_classes):
        super().__init__()

        self.channels, self.width, self.height = dimensions
        self.num_classes = num_classes

        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.channels, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(64, 256, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fully_connected_layers = nn.Sequential(
            nn.Linear(256, 32), nn.Linear(32, self.num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = nn.Flatten()(x)
        logits = self.fully_connected_layers(x)
        return logits


img_dimensions = (1, 28, 28)
model = QuickDrawNetwork_V1(img_dimensions, len(class_names))
num_of_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

epochs = config["epochs"]
loss_fn = get_torch_loss(config["loss"])
optimizer = get_torch_optimizer(
    config["optimizer"], model.parameters(), config["learning_rate"]
)
device = config["device"]

model.to(device)
for t in range(1):
    print(f"Training on : {device}")
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer, device)
    test(test_dataloader, model, loss_fn, device)
print("Done!")

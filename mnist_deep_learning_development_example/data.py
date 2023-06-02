from pathlib import Path
import struct
from array import array

import pytorch_lightning as pl
import numpy as np
import torch
import torchvision.transforms as T

class Dataset(torch.utils.data.Dataset):

    def __init__(self, images, labels, transform):
        super().__init__()
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.transform(self.images[idx], self.labels[idx])

class Transforms():
    def __init__(self):
        self.transform = T.ToTensor()

    def onehot(self, label):
        encoded_label = np.zeros(10)
        encoded_label[label] = 1
        return encoded_label

    def __call__(self, image, label):
        return self.transform(image), self.onehot(label)


class DataModule(pl.LightningDataModule):
    def prepare_data(self):
        train_val_images = self.read_image_data("train-images.idx3-ubyte")
        train_val_labels = self.read_labels("train-labels.idx1-ubyte")
        self.train_images = train_val_images[:int(0.9 * len(train_val_images))]
        self.train_labels = train_val_labels[:len(self.train_images)]

        self.val_images = train_val_images[len(self.train_images):]
        self.val_labels = train_val_labels[len(self.train_images):]

        self.test_images = self.read_image_data("t10k-images.idx3-ubyte")
        self.test_labels = self.read_labels("t10k-labels.idx1-ubyte")

    def setup(self, stage):
        self.train_dataset = Dataset(self.train_images, self.train_labels, Transforms())
        self.val_dataset = Dataset(self.val_images, self.val_labels, Transforms())
        self.test_dataset = Dataset(self.test_images, self.test_labels, Transforms())

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=100, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=100)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=100)

    def read_image_data(self, path):
        data_dir = Path(__file__).parent.parent / "mnist-dataset"
        with open(data_dir / path, "rb") as f:
            # IDX file format
            magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
            image_data = array("B", f.read())
        images = []
        for i in range(size):
            image = np.array(image_data[i * rows * cols:(i + 1) * rows * cols]).reshape(28, 28)
            images.append(image)
        return np.array(images)


    def read_labels(self, path):
        data_dir = Path(__file__).parent.parent / "mnist-dataset"
        with open(data_dir / path, "rb") as f:
            magic, size = struct.unpack(">II", f.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = np.array(array("B", f.read()))
        return labels

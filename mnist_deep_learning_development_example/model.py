import torch
import torch.nn as nn
import pytorch_lightning as pl

class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 1,  out_channels=9, kernel_size=(3, 3))
        self.norm1 = nn.BatchNorm2d(9)
        self.pool = nn.MaxPool2d((3, 3))
        self.conv2 = nn.Conv2d(in_channels = 9,  out_channels=5, kernel_size=(3, 3))
        self.norm2 = nn.BatchNorm2d(5)
        self.fc1 = nn.Linear(in_features=320, out_features=10)

    def forward(self, digit):
        out = nn.functional.relu(self.norm1(self.conv1(digit)))
        out = self.pool(nn.functional.relu(self.norm2(self.conv2(out))))
        out = self.fc1(out.flatten(1))
        return out

class Model(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("Model")
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        return parent_parser

    @classmethod
    def from_argparse_args(cls, *args, **kwargs):
        """Pass the ArgParser's args to the constructor."""
        return pl.utilities.argparse.from_argparse_args(cls, *args, **kwargs)


    def __init__(self, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.model = CNN()
        self.loss = torch.nn.functional.cross_entropy
        self.val_outputs = []
        self.test_outputs = []

    def forward(self, images):
        return self.model(images)

    def _step(self, stage, batch, batch_idx):
        images, y_true = batch
        y_pred = self(images)
        loss = self.loss(y_pred, y_true.long())
        self.log(f"loss/{stage}", loss, prog_bar=True)
        return {"loss": loss,
                "labels": y_true,
                "preds": torch.argmax(y_pred.detach(), axis=1)}

    def training_step(self, batch, batch_idx):
        return self._step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        self.val_outputs.append(self._step("val", batch, batch_idx))
        return self.val_outputs[-1]

    def test_step(self, batch, batch_idx):
        self.test_outputs.append(self._step("test", batch, batch_idx))
        return self.test_outputs[-1]

    def _epoch_end(self, stage, outputs):
        predictions = torch.hstack([o["preds"] for o in outputs])
        labels = torch.hstack([o["labels"] for o in outputs])
        total = len(labels)
        accuracy = sum(predictions == labels).item() / total * 100
        error_rate = sum(predictions != labels).item() / total * 100
        self.log_dict({f"accuracy/{stage}": accuracy,
                       f"error_rate/{stage}": error_rate}, prog_bar=True)

    def on_validation_epoch_end(self):
        return self._epoch_end("val", self.val_outputs)

    def on_test_epoch_end(self):
        return self._epoch_end("test", self.test_outputs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)

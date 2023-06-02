#!/usr/bin/env python3
from argparse import ArgumentParser

import pytorch_lightning as pl

from mnist_deep_learning_development_example.model import Model
from mnist_deep_learning_development_example.data import DataModule

if __name__ == "__main__":
    parser = ArgumentParser()
    parser = DataModule.add_argparse_args(parser)
    parser = Model.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    datamodule = DataModule()
    args, unknown = parser.parse_known_args()
    model = Model.from_argparse_args(args)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)

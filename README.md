# Deep Learning Development using PyTorch Lightning and Guild.ai

This repository provides all the necessary code to train a CNN for classifying MNIST images.
The source code shows how to develop a model using PyTorch Lightning and train it using guild.ai.

## Setup
### Install the dependencies

``` sh
poetry install
```

### Download the MNIST data set
``` sh
$ kaggle datasets download hojjatk/mnist-dataset 
$ unzip mnist-dataset.zip -d mnist-dataset
```
Adjust the `data_dir` in `data.py`, if necessary.

### Train the model using PyTorch Lightning

``` sh
python mnist_deep_learning_development_example/main.py --max_epochs=1
```

### Train the model using guild.ai

``` sh
guild run train max_epochs=1
```

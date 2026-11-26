# CNN on CIFAR-10 Dataset

## Overview

This notebook implements a Convolutional Neural Network (CNN) from scratch using PyTorch to classify images from the CIFAR-10 dataset. The model is trained and evaluated on 10 object categories, achieving around 68% test accuracy over 100 epochs.

## Dataset

The CIFAR-10 dataset is loaded from Hugging Face (`uoft-cs/cifar10`) and contains:

- 50,000 training images
- 10,000 test images
- Image size: 32x32 pixels, 3 color channels (RGB)
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Model Architecture

The model `CIFAR_MODEL_V0` is a custom CNN built with `torch.nn.Module` and consists of:

- Layer 1: Two Conv2d layers (3 -> 10 channels) with ReLU activations, followed by MaxPool2d
- Layer 2: Two Conv2d layers (10 -> 10 channels) with ReLU activations, followed by MaxPool2d
- Classifier: Flatten layer followed by a Linear layer (640 -> 10 output classes)

## Training Details

- Loss Function: CrossEntropyLoss
- Optimizer: Adam (learning rate = 0.001)
- Epochs: 100
- Batch Size: 32
- Device: CPU
- Accuracy Metric: torchmetrics Accuracy (multiclass, 10 classes)

## Results

After 100 epochs of training:

- Train Accuracy: ~74%
- Test Accuracy: ~68%

The model shows consistent improvement across epochs with training accuracy slightly higher than test accuracy, indicating mild overfitting.

## Dependencies

- Python 3.x
- PyTorch 2.9.1+cu130
- torchvision
- torchmetrics
- datasets (Hugging Face)
- matplotlib
- numpy
- tqdm

## Usage

Run the notebook cells sequentially:

1. Import libraries and verify GPU availability
2. Load the CIFAR-10 dataset via Hugging Face
3. Explore and visualize sample images
4. Define and instantiate the CNN model
5. Train and evaluate the model using the `train_test_model` function
6. Visualize model predictions on test samples

## File

- `CNN_CIFAR_10.ipynb` - Main notebook containing all code, training logs, and visualizations

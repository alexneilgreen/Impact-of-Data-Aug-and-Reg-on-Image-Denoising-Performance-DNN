# Image Denoising with Augmentation

This repository contains an image denoising model trained on multiple datasets (MNIST, CIFAR10, CIFAR100, STL10) with various augmentation techniques. The model is implemented using PyTorch, and the results of different training experiments are analyzed and visualized.

## Requirements

Ensure you have the following installed:

- Python 3.8+

To install all dependencies, run:

```bash
pip install -r requirements.txt
```

## Running the Training Script

The `Main.py` script trains the image denoising model with different augmentation techniques on different datasets.

### Arguments:

- `--experiment`: Name of the experiment (e.g., `base`, `gaussian_noise`, `all`)
- `--dataset`: Dataset to use (`MNIST`, `CIFAR10`, `CIFAR100`, `STL10`)
- `--epochs`: Number of training epochs (default: `50`)
- `--learning_rate`: Learning rate for training (default: `0.001`)

### Basic Training

To train the base model without augmentation on MNIST:

```bash
python Main.py --experiment base --dataset MNIST
```

or specify additional parameters:

```bash
python Main.py --experiment base --dataset MNIST --epochs 50 --learning_rate 0.001
```

### Training on Different Datasets

To train on CIFAR10:

```bash
python Main.py --experiment base --dataset CIFAR10 --epochs 50 --learning_rate 0.001
```

Other supported datasets: `CIFAR100`, `STL10`

### Training with Augmentations

You can train with specific augmentation techniques:

```bash
python Main.py --experiment gaussian_noise --dataset MNIST --epochs 50 --learning_rate 0.001
```

Available augmentations:

- `brightness`
- `color_jitter`
- `contrast`
- `cutout`
- `flipping`
- `gaussian_noise`
- `random_crop`
- `rotation`
- `scaling`
- `shearing`
- `custom_augmentation_1`

To run all augmentations sequentially:

```bash
python Main.py --experiment all --dataset CIFAR10 --epochs 50 --learning_rate 0.001
```

### Training with Regularizations

You can train with specific regularization techniques:

```bash
python Main.py --regVal L1 --L1 1e-3 --dataset MNIST --epochs 50 --learning_rate 0.001
```

Available regularizations:

- `L1`
- `L2`
- `DR`
- `ES`

To run all regularizations with default tunning sequentially:

```bash
python Main.py --regVal all --dataset CIFAR10 --epochs 50 --learning_rate 0.001
```

### Training with Augmentation and Regularization Combo

You can train:

```bash
python Main.py --experiment combo
```

## Analyzing Results

After training, navigate to the `Analyze/` directory. To analyze and visualize results, run the following commands in order:

```bash
cd Analyze
```

```bash
python AnalyzeAug.py
```

```bash
python AnalyzePreProcess.py
```

```bash
python AnalyzeReg.py
```

This script generates:

- Loss and accuracy plots per epoch for each experiment
- A comparison of test loss, test accuracy, and computation time across experiments
- CSV files summarizing experiment results

## Directory Structure

```
.
├── Main.py       # Training script
├── Model.py      # Denoising model architecture
├── Train.py      # Training logic
├── Analyze/      # Analysis and visualization scripts
├── Results/      # Directories for storing experiment results
├── requirements.txt # Dependencies
├── README.md     # Documentation
```

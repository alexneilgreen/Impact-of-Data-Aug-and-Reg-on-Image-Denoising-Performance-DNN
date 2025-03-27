# Image Denoising with Augmentation

This repository contains an image denoising model trained on the MNIST dataset with various augmentation techniques. The model is implemented using PyTorch, and the results of different training experiments are analyzed and visualized.

## Requirements

Ensure you have the following installed:

- Python 3.8+

To install all dependencies, run:

```bash
pip install -r requirements.txt
```

## Running the Training Script

The `Main.py` script trains the image denoising model with different augmentation techniques.

### Arguments:

- `--experiment`: Name of the experiment (e.g., `base`, `gaussian_noise`, `all`)
- `--epochs`: Number of training epochs (default: `10`)
- `--learning_rate`: Learning rate for training (default: `0.001`)

### Basic Training

To train the base model without augmentation:

```bash
python Main.py --experiment base
```

or

```bash
python Main.py --experiment base --epochs 10 --learning_rate 0.001
```

### Training with Augmentations

You can train with specific augmentation techniques:

```bash
python Main.py --experiment gaussian_noise --epochs 10 --learning_rate 0.001
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

To run all augmentations sequentially:

```bash
python Main.py --experiment all --epochs 10 --learning_rate 0.001
```

## Analyzing Results

After training, results are saved in the `Results/` directory. To analyze and visualize them, run:

```bash
python Analyze.py
```

This script generates:

- Loss and accuracy plots per epoch for each experiment
- A comparison of test loss, test accuracy, and computation time across experiments
- CSV files summarizing experiment results

## Directory Structure

```
.
├── Main.py       # Training script
├── Analyze.py    # Analysis and visualization script
├── Model.py      # Denoising model architecture
├── Train.py      # Training logic
├── Results/      # Directory for storing experiment results
├── requirements.txt # Dependencies
├── README.md     # Documentation
```

## Citation

If you use this code in your research, please cite it accordingly.

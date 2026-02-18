<div align="center">

# CNN Image Classifier — CIFAR-10

**A convolutional neural network built from scratch with PyTorch to classify 32×32 color images across 10 categories.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## Overview

This project implements a **complete deep learning pipeline** — from raw data to trained model — following the [official PyTorch CIFAR-10 tutorial](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html). The CNN classifies 32×32 RGB images into one of 10 classes:

`✈️ plane` · `🚗 car` · `🐦 bird` · `🐱 cat` · `🦌 deer` · `🐶 dog` · `🐸 frog` · `🐴 horse` · `🚢 ship` · `🚚 truck`

> **Key takeaway:** Demonstrates practical understanding of neural network design, training loops, loss optimization, and model evaluation.

---

## Architecture

The CNN follows a classic **LeNet-inspired** design with two convolutional feature extractors followed by three fully-connected classifiers:

```
Input (3×32×32 RGB)
  │
  ├─► Conv2d(3→6, 5×5) + ReLU + MaxPool(2×2)    →  6×14×14
  ├─► Conv2d(6→16, 5×5) + ReLU + MaxPool(2×2)   →  16×5×5
  │
  ├─► Flatten                                   →  400
  ├─► Linear(400→120) + ReLU
  ├─► Linear(120→84)  + ReLU
  └─► Linear(84→10)                             →  10 class logits
```

| Component | Details |
|-----------|---------|
| **Loss Function** | Cross-Entropy Loss |
| **Optimizer** | SGD (lr=0.001, momentum=0.9) |
| **Batch Size** | 4 |
| **Epochs** | 2 |
| **Device** | Auto-detects CUDA / Apple MPS / CPU |

---

## Results

After training for just **2 epochs**, the model achieves **~52% accuracy** on 10,000 unseen test images — significantly better than random chance (10%). 

| Class | Accuracy |
|-------|----------|
| Plane | ~66% |
| Car | ~83% |
| Bird | ~59% |
| Cat | ~29% |
| Deer | ~30% |
| Dog | ~45% |
| Frog | ~58% |
| Horse | ~63% |
| Ship | ~51% |
| Truck | ~42% |

> Training for more epochs and/or increasing the network width would improve these numbers further. Increasing and balancing the dataset might also increase the accuracy. 

---

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Setup & Run

```bash
# Clone the repository
git clone https://github.com/<your-username>/cnn-cifar10.git
cd cnn-cifar10

# Install dependencies
pip install -r requirements.txt

# Train and evaluate the model
python cifar10_cnn.py
```

The script will automatically:
1. Download the CIFAR-10 dataset (~170 MB, first run only)
2. Display a sample batch of training images
3. Train the CNN for 2 epochs (prints loss every 2,000 batches)
4. Save the model weights to `cifar_net.pth`
5. Evaluate overall and per-class accuracy on the test set

---

## Project Structure

```
cnn-cifar10/
├── cifar10_cnn.py       # Main script — training & evaluation pipeline
├── requirements.txt     # Python dependencies
├── .gitignore           # Ignores dataset, weights, caches
└── README.md            # You are here
```

**Generated at runtime:**
```
├── data/                # Downloaded CIFAR-10 dataset (gitignored)
└── cifar_net.pth        # Saved model weights (gitignored)
```

---

## Technologies & Skills Demonstrated

- **PyTorch** — tensor operations, `nn.Module` subclassing, autograd
- **Computer Vision** — convolutional neural networks, image normalization, data augmentation pipeline
- **Training Pipeline** — loss computation, backpropagation, SGD optimization, mini-batch processing
- **Model Persistence** — `state_dict` serialization and deserialization
- **GPU Acceleration** — automatic device selection (CUDA / MPS / CPU)
- **Data Visualization** — matplotlib for image grid rendering
- **Clean Code Practices** — docstrings, type documentation, constants extraction, modular design

---

## References

- [PyTorch CIFAR-10 Tutorial](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html) — Krizhevsky, 2009
- [PyTorch Documentation](https://docs.pytorch.org/docs/stable/)

---

<div align="center">

*Built as part of a research initiative at FUNAPE.*

</div>

"""
cifar10_cnn.py — CIFAR-10 Image Classifier using a Convolutional Neural Network

This script implements a complete CNN training and evaluation pipeline for the
CIFAR-10 dataset, following the official PyTorch tutorial:
https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

Steps:
    1. Load and normalize the CIFAR-10 training and test datasets.
    2. Define a Convolutional Neural Network (CNN).
    3. Define the loss function and optimizer.
    4. Train the network on the training data.
    5. Test the network on the test data.

Author:  Cristofer Silva
Project: CNN CIFAR-10 (FUNAPE)
Date:    2026-02-18
"""

# =============================================================================
# Imports
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# Constants
# =============================================================================

# Training hyper-parameters
BATCH_SIZE = 4
NUM_EPOCHS = 2
LEARNING_RATE = 0.001
MOMENTUM = 0.9

# DataLoader workers (set to 0 on Windows/macOS if you encounter BrokenPipeError)
NUM_WORKERS = 2

# Path where the trained model weights will be saved
MODEL_PATH = "./cifar_net.pth"

# CIFAR-10 class labels (order matches dataset indices 0-9)
CLASSES = (
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


# =============================================================================
# 1. Data Loading and Normalization
# =============================================================================

def get_data_loaders():
    """
    Download (if needed) and load the CIFAR-10 training and test sets.

    Images are converted to tensors and normalized from [0, 1] to [-1, 1]
    using mean=0.5 and std=0.5 for each of the three RGB channels.

    Returns:
        trainloader (DataLoader): Shuffled training data loader.
        testloader  (DataLoader): Non-shuffled test data loader.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Training set — 50,000 images
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
    )

    # Test set — 10,000 images
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform,
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    )

    return trainloader, testloader


# =============================================================================
# 2. CNN Architecture
# =============================================================================

class Net(nn.Module):
    """
    A simple Convolutional Neural Network for CIFAR-10 classification.

    Architecture:
        Input  → 3×32×32 RGB image
        Conv1  → 6 filters, 5×5 kernel  → 6×28×28
        Pool   → MaxPool 2×2            → 6×14×14
        Conv2  → 16 filters, 5×5 kernel → 16×10×10
        Pool   → MaxPool 2×2            → 16×5×5
        FC1    → 400 → 120
        FC2    → 120 → 84
        FC3    → 84  → 10 (output logits, one per class)
    """

    def __init__(self):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        # Pooling layer (shared between both convolutions)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully-connected (linear) layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """Forward pass: conv → relu → pool → conv → relu → pool → fc layers."""
        x = self.pool(F.relu(self.conv1(x)))   # (batch, 6, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))   # (batch, 16, 5, 5)
        x = torch.flatten(x, 1)                # Flatten all dims except batch
        x = F.relu(self.fc1(x))                # (batch, 120)
        x = F.relu(self.fc2(x))                # (batch, 84)
        x = self.fc3(x)                        # (batch, 10)  — raw logits
        return x


# =============================================================================
# Helper Utilities
# =============================================================================

def imshow(img):
    """
    Display a torchvision image grid.

    The image is first un-normalized from [-1, 1] back to [0, 1],
    then converted from (C, H, W) to (H, W, C) for matplotlib.

    Args:
        img (Tensor): A tensor image, typically from `torchvision.utils.make_grid`.
    """
    img = img / 2 + 0.5  # Un-normalize: [-1, 1] → [0, 1]
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()


def get_device():
    """
    Select the best available compute device.

    Priority: CUDA GPU → Apple MPS → CPU.

    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    return device


# =============================================================================
# 3 & 4. Training
# =============================================================================

def train(net, trainloader, device):
    """
    Train the network for NUM_EPOCHS using Cross-Entropy loss and SGD.

    Loss statistics are printed every 2,000 mini-batches.

    Args:
        net         (Net):        The CNN model (already on `device`).
        trainloader (DataLoader): Training data loader.
        device      (torch.device): Compute device.
    """
    # --- Step 3: Loss function and optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    # --- Step 4: Training loop ---
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            # Move inputs and labels to the target device
            inputs, labels = data[0].to(device), data[1].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass → compute loss → backward pass → update weights
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Accumulate and print running loss every 2,000 mini-batches
            running_loss += loss.item()
            if i % 2000 == 1999:
                avg_loss = running_loss / 2000
                print(f"[Epoch {epoch + 1}, Batch {i + 1:5d}] loss: {avg_loss:.3f}")
                running_loss = 0.0

    print("Finished Training")

    # Save the trained model weights
    torch.save(net.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


# =============================================================================
# 5. Testing / Evaluation
# =============================================================================

def show_sample_predictions(net, testloader, device):
    """
    Display a batch of test images alongside their ground-truth and predicted labels.

    Args:
        net        (Net):        The trained CNN model.
        testloader (DataLoader): Test data loader.
        device     (torch.device): Compute device.
    """
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # Show the images
    imshow(torchvision.utils.make_grid(images))
    print("GroundTruth: ", " ".join(f"{CLASSES[labels[j]]:5s}" for j in range(BATCH_SIZE)))

    # Predict
    outputs = net(images.to(device))
    _, predicted = torch.max(outputs, 1)

    print("Predicted:   ", " ".join(f"{CLASSES[predicted[j]]:5s}" for j in range(BATCH_SIZE)))


def evaluate_overall_accuracy(net, testloader, device):
    """
    Compute and print the overall accuracy on the full test set (10,000 images).

    Args:
        net        (Net):        The trained CNN model.
        testloader (DataLoader): Test data loader.
        device     (torch.device): Compute device.
    """
    correct = 0
    total = 0

    # Disable gradient computation — not needed for inference
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"\nAccuracy of the network on the 10,000 test images: {accuracy:.1f} %")


def evaluate_per_class_accuracy(net, testloader, device):
    """
    Compute and print accuracy for each of the 10 CIFAR-10 classes.

    Args:
        net        (Net):        The trained CNN model.
        testloader (DataLoader): Test data loader.
        device     (torch.device): Compute device.
    """
    correct_pred = {classname: 0 for classname in CLASSES}
    total_pred = {classname: 0 for classname in CLASSES}

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)

            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[CLASSES[label]] += 1
                total_pred[CLASSES[label]] += 1

    # Print per-class results
    print("\nPer-class accuracy:")
    print("-" * 30)
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f"  {classname:5s}  →  {accuracy:.1f} %")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """
    Orchestrate the full pipeline:
        1. Load data
        2. Build model & move to device
        3–4. Train the model
        5. Evaluate on test data
    """
    # ---- Device selection ----
    device = get_device()

    # ---- Step 1: Data loading ----
    trainloader, testloader = get_data_loaders()

    # Show a batch of training images (optional visualization)
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images))
    print("Sample labels: " + " ".join(f"{CLASSES[labels[j]]:5s}" for j in range(BATCH_SIZE)))

    # ---- Step 2: Instantiate the CNN ----
    net = Net()
    net.to(device)

    # ---- Steps 3 & 4: Train ----
    train(net, trainloader, device)

    # ---- Step 5: Test ----
    # Reload the saved model to demonstrate save/load workflow
    net_eval = Net()
    net_eval.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    net_eval.to(device)

    show_sample_predictions(net_eval, testloader, device)
    evaluate_overall_accuracy(net_eval, testloader, device)
    evaluate_per_class_accuracy(net_eval, testloader, device)


if __name__ == "__main__":
    main()

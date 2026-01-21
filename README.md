# PyTorch MNIST Classification (MLP vs CNN)
This repository is a **beginner-friendly yet well-structured PyTorch project** for handwritten digit classification on the **MNIST dataset**.

The project starts with a simple **MLP (Multi-Layer Perceptron)** and then upgrades to a **CNN (Convolutional Neural Network)**, demonstrating **why CNNs outperform MLPs on image tasks** through both quantitative results and training curves.

## Features

- Implemented using pure PyTorch

- Clean and modular project structure

- Supports CPU / GPU

- Automatic training loss & test accuracy visualization

- Clear comparison between MLP and CNN

- Reproducible and easy to extend

## Project Structure

```text

pytorch_mnist_project/
├── MLP/
│   ├── model.py        # MLP model definition
│   ├── train.py        # Training loop
│   ├── dataset.py      # MNIST dataloader
│   └── utils.py        # Evaluation function
│
├── CNN/
│   ├── model.py        # CNN model definition
│   ├── train.py        # Training + auto plotting
│   ├── dataset.py      # MNIST dataloader
│   └── utils.py        # Evaluation function
│
└── README.md

```
## Getting Started
### 1.Install Dependencies

```bash
pip install torch torchvision matplotlib
```

### 2.Train the CNN Model

```bash
cd CNN
python train.py
```

During training, the script will:

Print **loss and test accuracy** for each epoch

Automatically plot:

- Training Loss Curve

- Test Accuracy Curve

- Experimental Results

## CNN Performance on MNIST
Epoch	Test Accuracy

- 1 ·········· 98.46%

- 5 ·········· 99.19%

- 10 ········· 99.04%

- 20 ········· 99.2%

The CNN converges faster, more stably, and achieves higher accuracy than the MLP.

## Why CNN Outperforms MLP

- Local receptive fields capture **spatial patterns**

- **Parameter sharing** improves generalization

- **Hierarchical feature learning** (edges → strokes → digits)

- **Stronger inductive bias for image data**

- **CNNs embed prior knowledge about image structure, while MLPs treat pixels as independent features.**

## Possible Improvements

- Add data augmentation

- Save and load best checkpoints

- Compare with classic architectures (LeNet, ResNet)

- Visualize learned convolutional filters

## Learning Purpose

This project is designed for:

**PyTorch beginners**

Students learning **deep learning**

Anyone who wants a clean MNIST baseline project

**It emphasizes understanding over complexity.**

## 📄 License

MIT License

## Author

**Roxy**

Thanks for checking out this project!  
Hope you find it useful and fun to explore.

Learn more about me:  
👉 https://roxy5201314.github.io/













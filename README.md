# Simple GEP

A PyTorch implementation of Gradient Embedding Perturbation (GEP), a differential privacy method for deep learning. This implementation is based on the paper ["Gradient Embedding Perturbation for Differentially Private Learning"](https://github.com/dayu11/Differentially-Private-Deep-Learning).

## Overview

GEP is a novel approach to differential privacy in deep learning that:
- Projects high-dimensional gradients into a lower-dimensional space
- Adds noise more efficiently than traditional DP-SGD
- Utilizes public data to improve privacy-utility trade-offs
- Supports multiple embedding techniques for gradient dimension reduction

## Features

### Privacy Methods
- DP-SGD (Differential Privacy Stochastic Gradient Descent)
- GEP with configurable parameters
- Dynamic noise scaling options
- Multiple gradient clipping strategies:
  - Median-based
  - Value-based
  - Max-based

### Embedding Techniques
- SVD (Singular Value Decomposition)
- Kernel PCA with multiple kernels:
  - RBF (Radial Basis Function)
  - Linear
  - Polynomial
  - Sigmoid
  - Cosine

### Training Features
- Support for auxiliary public datasets
- Customizable basis element count
- Gradient history management
- Checkpoint saving/loading
- WandB integration for experiment tracking

## Requirements

- Python 3.11+
- PyTorch
- CUDA support (recommended)

## Installation

1. Clone the repository:
    ```bash
    git clone [https://github.com/moshebeutel/SimpleGEP.git](https://github.com/moshebeutel/SimpleGEP.git) && cd SimpleGEP
    ```

2. Install the project in a virtual environment:
    ```bash
    python -m venv venv source venv/bin/activate
    pip install -e .
    ```
    OR
    ```bash
    poetry lock
    poetry install
    ```
## Usage

### Basic Training
```bash
python simplegep/main.py
--dataset cifar10
--model_name tiny_cifar_net_4
--batchsize 256
--num_epochs 30
--dp_method dp_sgd # or  gep
```
when using poetry type ```poetry run python simplegep/main.py ...```
### Sweep

```bash
python simplegep/sweepers/sweep.py --dp_method gep --eps 1
``` 
when using poetry type ```poetry run python simplegep/sweepers/sweep.py ...```



# Neural Network from Scratch with NumPy

This project implements a simple neural network **from scratch using only NumPy** to solve the classic **XOR problem**.

The goal of this project is to understand the fundamental mechanics of neural networks without using high-level frameworks.

## Concepts Implemented

* Forward Propagation
* Backpropagation
* Gradient Descent
* Sigmoid Activation Function
* Matrix Operations using NumPy

These are the core ideas behind modern deep learning systems.

## Architecture

Neural Network Structure:

Input Layer: 2 neurons
Hidden Layer: 2 neurons
Output Layer: 1 neuron

```
Input (2)
   ↓
Hidden Layer (2)
   ↓
Output (1)
```

## Dataset

The model learns the XOR logic gate:

| x1 | x2 | Output |
| -- | -- | ------ |
| 0  | 0  | 0      |
| 0  | 1  | 1      |
| 1  | 0  | 1      |
| 1  | 1  | 0      |

## Example Output

After training:

```
[[0.02]
 [0.97]
 [0.96]
 [0.03]]
```

The model successfully learns the XOR pattern.

## Requirements

* Python 3
* NumPy

Install dependencies:

```
pip install numpy
```

## Run the Project

```
python nn.py
```

## Purpose

This project is built for educational purposes to understand how neural networks work internally before using frameworks like PyTorch or TensorFlow.

## Technologies Used

* Python
* NumPy

## Future Improvements

* Add loss visualization
* Implement more activation functions
* Train on larger datasets
* Build a modular neural network class

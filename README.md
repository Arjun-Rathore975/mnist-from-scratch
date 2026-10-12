# MNIST Neural Network from Scratch

A handwritten digit classifier built using only **NumPy** — no TensorFlow, no PyTorch. Every part of the network (forward propagation, backpropagation, gradient descent) is implemented by hand to understand how neural networks actually learn.

![Example prediction](example.png)

## How it works

A simple two-layer neural network trained on the [MNIST](https://www.kaggle.com/competitions/digit-recognizer) dataset of 28×28 grayscale digits.

| Layer | Size | Activation |
|---|---|---|
| Input | 784 (28×28 pixels) | — |
| Hidden | 10 | ReLU |
| Output | 10 (digits 0–9) | Softmax |

**Training loop:**
1. **Forward prop** – pass images through the network to get a probability for each digit
2. **Backprop** – compare predictions to the true labels (one-hot encoded) and compute the gradient of the loss for every weight and bias
3. **Update** – nudge every parameter in the direction that reduces the error (gradient descent, learning rate 0.1)
4. Repeat for 500 iterations

## Results

- Dev set accuracy: **~XX%** on 1,000 held-out images

## Running it

The notebook is set up for [Kaggle](https://www.kaggle.com/):

1. Open a new Kaggle notebook and upload `mnist-neural-network.ipynb`
2. Click **Add Input** and attach the **Digit Recognizer** competition
3. Click **Run All**

To run locally instead, download `train.csv` from the [competition page](https://www.kaggle.com/competitions/digit-recognizer/data) and change the path in the `pd.read_csv(...)` cell.

**Requirements:** `numpy`, `pandas`, `matplotlib`

## Credits

Based on Samson Zhang's video [Building a neural network FROM SCRATCH](https://www.youtube.com/watch?v=w8yWXqWQYmU), with fixes to softmax numerical stability and the bias gradients.

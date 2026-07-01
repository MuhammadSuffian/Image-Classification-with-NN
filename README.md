# Fashion-MNIST Image Classification with a Neural Network

A simple feedforward neural network built with TensorFlow/Keras to classify clothing images from the Fashion-MNIST dataset.

## Overview

This project trains a fully-connected (dense) neural network to recognize 10 categories of clothing items  such as t-shirts, trousers, and sneakers  from 28x28 grayscale images. It's a foundational deep learning exercise demonstrating the end-to-end workflow of loading data, building a model, training it, and generating predictions.

## Dataset

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) is a drop-in replacement for the classic MNIST digits dataset, consisting of:

- **60,000** training images
- **10,000** test images
- **28x28** grayscale pixel images
- **10 classes**: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

The dataset is loaded directly via `keras.datasets.fashion_mnist`, so no manual download is required.

## Model Architecture

A `Sequential` model with the following layers:

| Layer | Type | Units | Activation |
|---|---|---|---|
| 1 | Flatten | 28x28 → 784 | — |
| 2 | Dense | 300 | ReLU |
| 3 | Dense | 100 | ReLU |
| 4 | Dense | 10 | Softmax |

**Compilation settings:**
- Loss: `sparse_categorical_crossentropy`
- Optimizer: `sgd` (Stochastic Gradient Descent)
- Metric: `accuracy`

## Workflow

1. **Load data** — Fashion-MNIST training and test sets via Keras.
2. **Explore data** — Visualize a sample image and inspect dataset shapes.
3. **Preprocess** — Normalize pixel values to the `[0, 1]` range and split off a 5,000-sample validation set from the training data.
4. **Build model** — Define the dense network architecture.
5. **Train** — Fit the model for 30 epochs, validating on the held-out set.
6. **Predict** — Run inference on test samples and extract predicted class labels using `argmax`.

## Requirements

```
tensorflow
numpy
matplotlib
```

Install with:

```bash
pip install tensorflow numpy matplotlib
```

## Usage

1. Clone this repository.
2. Open `Image_Classification_with_NN.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
3. Run all cells sequentially to train the model and view predictions.

## Results

The model outputs a probability distribution across the 10 clothing classes for each input image, with the predicted label taken as the class with the highest probability (`np.argmax`).

## Possible Improvements

- Add a test set evaluation step (`model.evaluate`) to report final accuracy and loss.
- Experiment with a CNN architecture for improved accuracy over the dense network.
- Try the Adam optimizer for potentially faster convergence.
- Add early stopping and learning rate scheduling.
- Visualize training/validation accuracy and loss curves.

## License

This project is open source and available under the [MIT License](LICENSE).

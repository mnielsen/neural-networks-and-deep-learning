"""
mnist_autoencoder
~~~~~~~~~~~~~~~~~

Implements an autoencoder for the MNIST data, and plots the
autoencoder's output for the first ten digits in the MNIST test set.
"""

# My Libraries
from backprop2 import Network
import mnist_loader 

# Third-party libraries
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Constants
HIDDEN_UNITS=30 # size of the hidden layer in the autoencoder

def main():
    # Do the training
    training_data, test_inputs, actual_test_results = \
        mnist_loader.load_data_nn()
    autoencoder_training_data = [(x, x) for x, _ in training_data]
    net = Network([784, HIDDEN_UNITS, 784])
    net.SGD(autoencoder_training_data, 3, 10, 0.01, 0.05)
    # Plot the first ten outputs
    fig = plt.figure()
    ax = fig.add_subplot(111)
    images_in = [autoencoder_training_data[j][0].reshape(-1, 28)
                 for j in range(10)]
    images_out = [net.feedforward(autoencoder_training_data[j][0]).\
                      reshape(-1, 28) for j in range(10)]
    image_in = np.concatenate(images_in, axis=1)
    image_out = np.concatenate(images_out, axis=1)
    image = np.concatenate([image_in, image_out])
    ax.matshow(image, cmap = matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.show()

if __name__ == "__main__":
    main()

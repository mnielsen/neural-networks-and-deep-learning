"""
mnist_autoencoder
~~~~~~~~~~~~~~~~~

Implements an autoencoder for the MNIST data.  The program can do two
things: (1) plot the autoencoder's output for the first ten images in
the MNIST test set; and (2) use the autoencoder to build a classifier.
The program is a quick-and-dirty hack --- we'll do things in a more
systematic way in the module ``deep_autoencoder``.
"""

# My Libraries
from backprop2 import Network
import mnist_loader 

# Third-party libraries
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def autoencoder_results(hidden_units):
    """
    Train an autoencoder using the MNIST training data and plot the
    results when the first ten MNIST test images are passed through
    the autoencoder.
    """
    training_data, test_inputs, actual_test_results = \
        mnist_loader.load_data_nn()
    net = train_autoencoder(hidden_units, training_data)
    plot_test_results(net, test_inputs, actual_test_results)

def train_autoencoder(hidden_units, training_data):
    "Return a trained autoencoder."
    autoencoder_training_data = [(x, x) for x, _ in training_data]
    net = Network([784, hidden_units, 784])
    net.SGD(autoencoder_training_data, 3, 10, 0.01, 0.05)
    return net

def plot_test_results(net, test_inputs, actual_test_results):
    """
    Plot the results after passing the first ten test MNIST digits through
    the autoencoder ``net``."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    images_in = [test_inputs[j].reshape(-1, 28) for j in range(10)]
    images_out = [net.feedforward(test_inputs[j]).reshape(-1, 28) 
                  for j in range(10)]
    image_in = np.concatenate(images_in, axis=1)
    image_out = np.concatenate(images_out, axis=1)
    image = np.concatenate([image_in, image_out])
    ax.matshow(image, cmap = matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.show()

def classifier(hidden_units):
    """
    Train an autoencoder using the MNIST training data, and then use
    the autoencoder to create a classifier with a single hidden layer.
    """
    training_data, test_inputs, actual_test_results = \
        mnist_loader.load_data_nn()
    net_ae = train_autoencoder(hidden_units, training_data)
    net_c = Network([784, hidden_units, 10])
    net_c.biases = net_ae.biases[:2]+[np.random.randn(10, 1)/np.sqrt(10)]
    net_c.weights = net_ae.weights[:2]+\
        [np.random.randn(10, hidden_units)/np.sqrt(10)]
    net_c.SGD(training_data, 3, 10, 0.01, 0.05)
    print net_c.evaluate(test_inputs, actual_test_results)
    return net_c

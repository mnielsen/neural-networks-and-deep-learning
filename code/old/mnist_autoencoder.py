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
    plot_test_results(net, test_inputs)

def train_autoencoder(hidden_units, training_data):
    "Return a trained autoencoder."
    autoencoder_training_data = [(x, x) for x, _ in training_data]
    net = Network([784, hidden_units, 784])
    net.SGD(autoencoder_training_data, 6, 10, 0.01, 0.05)
    return net

def plot_test_results(net, test_inputs):
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

def classifier(hidden_units, n_unlabeled_inputs, n_labeled_inputs):
    """
    Train a semi-supervised classifier.  We begin with pretraining,
    creating an autoencoder which uses ``n_unlabeled_inputs`` from the
    MNIST training data.  This is then converted into a classifier
    which is fine-tuned using the ``n_labeled_inputs``.

    For comparison a classifier is also created which does not make
    use of the unlabeled data.
    """
    training_data, test_inputs, actual_test_results = \
        mnist_loader.load_data_nn()
    print "\nUsing pretraining and %s items of unlabeled data" %\
        n_unlabeled_inputs
    net_ae = train_autoencoder(hidden_units, training_data[:n_unlabeled_inputs])
    net_c = Network([784, hidden_units, 10])
    net_c.biases = net_ae.biases[:1]+[np.random.randn(10, 1)/np.sqrt(10)]
    net_c.weights = net_ae.weights[:1]+\
        [np.random.randn(10, hidden_units)/np.sqrt(10)]
    net_c.SGD(training_data[-n_labeled_inputs:], 300, 10, 0.01, 0.05)
    print "Result on test data: %s / %s" % (
        net_c.evaluate(test_inputs, actual_test_results), len(test_inputs))
    print "Training a network with %s items of training data" % n_labeled_inputs
    net = Network([784, hidden_units, 10])
    net.SGD(training_data[-n_labeled_inputs:], 300, 10, 0.01, 0.05)
    print "Result on test data: %s / %s" % (
        net.evaluate(test_inputs, actual_test_results), len(test_inputs))
    return net_c

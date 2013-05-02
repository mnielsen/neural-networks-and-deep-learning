"""
deep_autoencoder
~~~~~~~~~~~~~~~~

A module which implements deep autoencoders.  
"""

#### Libraries
# Standard library
import random

# My libraries
from backprop2 import Network, sigmoid_vec

# Third-party libraries
import numpy as np


def plot_helper(x):
    import matplotlib
    import matplotlib.pyplot as plt
    x = np.reshape(x, (-1, 28))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(x, cmap = matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.show()


class DeepAutoencoder(Network):

    def __init__(self, layers):
        """
        The list ``layers`` specifies the sizes of the nested
        autoencoders.  For example, if ``layers`` is [50, 20, 10] then
        the deep autoencoder will be a neural network with layers of
        size [50, 20, 10, 20, 50]."""
        self.layers = layers
        Network.__init__(self, layers+layers[-2::-1])

    def train(self, training_data, epochs, mini_batch_size, eta,
              lmbda):
        """
        Train the DeepAutoencoder.  The ``training_data`` is a list of
        training inputs, ``x``, ``mini_batch_size`` is a single
        positive integer, and ``epochs``, ``eta``, ``lmbda`` are lists
        of parameters, with the different list members corresponding
        to the different stages of training.  For example, ``eta[0]``
        is the learning rate used for the first nested autoencoder,
        ``eta[1]`` is the learning rate for the second nested
        autoencoder, and so on.  ``eta[-1]`` is the learning rate used
        for the final stage of fine-tuning.
        """
        print "\nTraining a %s deep autoencoder" % (
            "-".join([str(j) for j in self.sizes]),)
        training_data = double(training_data)
        cur_training_data = training_data[::]
        for j in range(len(self.layers)-1):
            print "\nTraining the %s-%s-%s nested autoencoder" % (
                self.layers[j], self.layers[j+1], self.layers[j])
            print "%s epochs, mini-batch size %s, eta = %s, lambda = %s" % (
                epochs[j], mini_batch_size, eta[j], lmbda[j])
            self.train_nested_autoencoder(
                j, cur_training_data, epochs[j], mini_batch_size, eta[j],
                lmbda[j])
            cur_training_data = [
                (sigmoid_vec(np.dot(net.weights[0], x)+net.biases[0]),)*2
                for (x, _) in cur_training_data]
        print "\nFine-tuning network weights with backpropagation"
        print "%s epochs, mini-batch size %s, eta = %s, lambda = %s" % (
                epochs[-1], mini_batch_size, eta[-1], lmbda[-1])
        self.SGD(training_data, epochs[-1], mini_batch_size, eta[-1],
                 lmbda[-1])

    def train_nested_autoencoder(
        self, j, encoded_training_data, epochs, mini_batch_size, eta, lmbda):
        """
        Train the nested autoencoder that starts at layer ``j`` in the
        deep autoencoder.  Note that ``encoded_training_data`` is a
        list with entries of the form ``(x, x)``, where the ``x`` are
        encoded training inputs for layer ``j``."""
        net = Network([self.layers[j], self.layers[j+1], self.layers[j]])
        net.biases[0] = self.biases[j]
        net.biases[1] = self.biases[-j-1]
        net.weights[0] = self.weights[j]
        net.weights[1] = self.weights[-j-1]
        net.SGD(encoded_training_data, epochs, mini_batch_size, eta, lmbda)
        self.biases[j] = net.biases[0]
        self.biases[-j-1] = net.biases[1]
        self.weights[j] = net.weights[0]
        self.weights[-j-1] = net.weights[1]

    def train_nested_autoencoder_repl(
        self, j, training_data, epochs, mini_batch_size, eta, lmbda):
        """
        This is a convenience method that can be used from the REPL to
        train the nested autoencoder that starts at level ``j`` in the
        deep autoencoder.  Note that ``training_data`` is the input
        data for the first layer of the network, and is a list of
        entries ``x``."""
        self.train_nested_autoencoder(
            j, 
            double(
                [self.feedforward(x, start=0, end=j) for x in training_data]),
            epochs, mini_batch_size, eta, lmbda)

    def feature(self, j, k):
        """
        Return the output if neuron number ``k`` in layer ``j`` is
        activated, and all others are not active.  """
        a = np.zeros((self.sizes[j], 1))
        a[k] = 1.0
        return self.feedforward(a, start=j, end=self.num_layers)

def double(l):
    return [(x, x) for x in l]


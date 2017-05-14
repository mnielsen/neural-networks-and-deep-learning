"""
deep_learning
~~~~~~~~~~~~~

Module to do deep learning.  Most of the functionality needed is
already in the ``backprop2`` and ``deep_autoencoder`` modules, but
this adds convenience functions to help in doing things like unrolling
deep autoencoders, and adding and training a classifier layer."""

# My Libraries
from backprop2 import Network
from deep_autoencoder import DeepAutoencoder

def unroll(deep_autoencoder):
    """
    Return a Network that contains the compression stage of the
    ``deep_autoencoder``."""
    net = Network(deep_autoencoder.layers)
    net.weights = deep_autoencoder.weights[:len(deep_autoencoder.layers)-1]
    net.biases = deep_autoencoder.biases[:len(deep_autoencoder.layers)-1]
    return net

def add_classifier_layer(net, num_outputs):
    """
    Return the Network ``net``, but with an extra layer containing
    ``num_outputs`` neurons appended."""
    net_classifier = Network(net.sizes+[num_outputs])
    net_classifier.weights[:-1] = net.weights
    net_classifier.biases[:-1] = net.biases
    return net_classifier

def SGD_final_layer(
    self, training_data, epochs, mini_batch_size, eta, lmbda):
    """
    Run SGD on the final layer of the Network ``self``.  Note that
    ``training_data`` is the input to the whole Network, not the
    encoded training data input to the final layer. 
    """
    encoded_training_data = [
        (self.feedforward(x, start=0, end=self.num_layers-2), y) 
        for x, y in training_data]
    net = Network(self.sizes[-2:])
    net.biases[0] = self.biases[-1]
    net.weights[0] = self.weights[-1]
    net.SGD(encoded_training_data, epochs, mini_batch_size, eta, lmbda)
    self.biases[-1] = net.biases[0]
    self.weights[-1] = net.weights[0]


# Add the SGD_final_layer method to the Network class
Network.SGD_final_layer = SGD_final_layer

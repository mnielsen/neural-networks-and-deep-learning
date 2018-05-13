# -*- coding: utf-8 -*-

import mnist_loader
from network import Network

# Loading the MNIST data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()


# Set up a Network with 30 hidden neurons
net = Network([784, 30, 10])

# Use stochastic gradient descent to learn from the MNIST training_data over 
# 30 epochs, with a mini-batch size of 10, and a learning rate of η = 3.0
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
"""
overfitting
~~~~~~~~~~~

Plot graphs to illustrate the problem of overfitting.  Must be imported
as a module, and run with overfitting.main(filename, lmbda).
"""

# Standard library
import imp
import json
import sys

# My library
sys.path.append('../code/')
import mnist_loader
import network2

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np


# Number of epochs to train for 
NUM_EPOCHS = 100

# Make results more easily reproducible
import random
random.seed(12345678)


def main(filename, lmbda=0.0):
    """filename is the name of the file where the results will be stored.
    lmbda is the regularization parameter."""
    run_network(filename, lmbda)
    make_plots(filename)
                       
def run_network(filename, lmbda=0.0):
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost())
    test_cost, test_accuracy, training_cost, training_accuracy \
        = net.SGD(training_data[:1000], NUM_EPOCHS, 10, 0.05,
                  evaluation_data=test_data, lmbda = lmbda,
                  monitor_evaluation_cost=True, 
                  monitor_evaluation_accuracy=True, 
                  monitor_training_cost=True, 
                  monitor_training_accuracy=True)
    f = open(filename, "w")
    json.dump([test_cost, test_accuracy, training_cost, training_accuracy], f)
    f.close()

def make_plots(filename):
    f = open(filename, "r")
    test_cost, test_accuracy, training_cost, training_accuracy \
        = json.load(f)
    f.close()
    plot_training_cost(training_cost)
    plot_test_accuracy(test_accuracy)
    plot_test_cost(test_cost)
    plot_training_accuracy(training_accuracy)

def plot_training_cost(training_cost):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(40, NUM_EPOCHS, 1), training_cost[40:NUM_EPOCHS])
    ax.set_xlim([40, NUM_EPOCHS])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Cost on the training data')
    plt.show()

def plot_test_accuracy(test_accuracy):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(40, NUM_EPOCHS, 1), 
            [accuracy/100.0 for accuracy in test_accuracy[40:NUM_EPOCHS]])
    ax.set_xlim([40, NUM_EPOCHS])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Accuracy (%) on the test data')
    plt.show()

def plot_test_cost(test_cost):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(0, NUM_EPOCHS, 1), test_cost[0:NUM_EPOCHS])
    ax.set_xlim([0, NUM_EPOCHS])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Cost on the test data')
    plt.show()

def plot_training_accuracy(training_accuracy):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(0, NUM_EPOCHS, 1), 
            [accuracy/10.0 for accuracy in training_accuracy])
    ax.set_xlim([0, NUM_EPOCHS])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Accuracy (%) on the training data')
    plt.show()

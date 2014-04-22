"""
overfitting
~~~~~~~~~~~

Plot graphs to illustrate the problem of overfitting.  
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

# Make results more easily reproducible
import random
random.seed(12345678)


def main(filename, num_epochs, lmbda=0.0):
    """``filename`` is the name of the file where the results will be
    stored.  ``num_epochs`` is the number of epochs to train for.
    ``lmbda`` is the regularization parameter.

    """
    run_network(filename, num_epochs, lmbda)
    make_plots(filename, num_epochs)
                       
def run_network(filename, num_epochs, lmbda=0.0):
    """Train the network for ``num_epochs``, and store the results in
    ``filename``.  Those results can later be used by ``make_plots``.
    Note that the results are stored to disk in large part because
    it's convenient not to have to ``run_network`` each time we want
    to make a plot (it's slow).

    """
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost())
    net.large_weight_initializer()
    test_cost, test_accuracy, training_cost, training_accuracy \
        = net.SGD(training_data[:1000], num_epochs, 10, 0.05,
                  evaluation_data=test_data, lmbda = lmbda,
                  monitor_evaluation_cost=True, 
                  monitor_evaluation_accuracy=True, 
                  monitor_training_cost=True, 
                  monitor_training_accuracy=True)
    f = open(filename, "w")
    json.dump([test_cost, test_accuracy, training_cost, training_accuracy], f)
    f.close()

def make_plots(filename, num_epochs):
    """Load the results from ``filename``, and generate the corresponding
    plots."""
    f = open(filename, "r")
    test_cost, test_accuracy, training_cost, training_accuracy \
        = json.load(f)
    f.close()
    plot_training_cost(training_cost, num_epochs)
    plot_test_accuracy(test_accuracy, num_epochs)
    plot_test_cost(test_cost, num_epochs)
    plot_training_accuracy(training_accuracy, num_epochs)

def plot_training_cost(training_cost, num_epochs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(150, num_epochs, 1), training_cost[150:num_epochs],
            color='#2A6EA6')
    ax.set_xlim([150, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Cost on the training data')
    plt.show()

def plot_test_accuracy(test_accuracy, num_epochs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(150, num_epochs, 1), 
            [accuracy/100.0 for accuracy in test_accuracy[150:num_epochs]],
            color='#2A6EA6')
    ax.set_xlim([150, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Accuracy (%) on the test data')
    plt.show()

def plot_test_cost(test_cost, num_epochs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(0, num_epochs, 1), test_cost[0:num_epochs],
            color='#2A6EA6')
    ax.set_xlim([0, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Cost on the test data')
    plt.show()

def plot_training_accuracy(training_accuracy, num_epochs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(0, num_epochs, 1), 
            [accuracy/10.0 for accuracy in training_accuracy[0:num_epochs]],
            color='#2A6EA6')
    ax.set_xlim([0, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Accuracy (%) on the training data')
    plt.show()

if __name__ == "__main__":
    filename = raw_input("Enter a file name: ")
    num_epochs = float(raw_input(
        "Enter a value for the number of epochs to run for: "))
    lmbda = float(raw_input(
        "Enter a value for the regularization parameter, lambda: "))
    main(filename, num_epochs, lmbda)

"""more_data
~~~~~~~~~~~~

Plot a graph to illustrate the performance of MNIST when different
size training sets are used.

"""

# Standard library
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

# The sizes to use for the different training sets
SIZES = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000] 

def main():
    run_networks()
    make_plot()
                       
def run_networks():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost())
    accuracies = []
    for size in SIZES:
        print "Training with training data of size "+size+"\n"
        net.large_weight_initializer()
        num_epochs = 1500000 / size 
        net.SGD(training_data[:size], num_epochs, 10, 0.05, lmbda = 0.001)
        accuracy = net.accuracy(validation_data) / 100.0
        print "Accuracy was "+accuracy+" %"
        accuracies.append(accuracy)
    f = open("more_data.json", "w")
    json.dump(accuracies, f)
    f.close()

def make_plot():
    f = open("more_data.json", "r")
    accuracies = json.load(f)
    f.close()
    make_linear_plot(accuracies)
    make_log_plot(accuracies)

def make_linear_plot(accuracies):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(SIZES, accuracies, color='#2A6EA6')
    ax.plot(SIZES, accuracies, "o", color="#FFA933")
    ax.set_xlim(0, 50000)
    ax.set_ylim(60, 100)
    ax.grid(True)
    ax.set_xlabel('Training set size')
    ax.set_title('Accuracy (%) on the validation data')
    plt.show()

def make_log_plot(accuracies):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(SIZES, accuracies, color='#2A6EA6')
    ax.plot(SIZES, accuracies, "o", color="#FFA933")
    ax.set_xlim(100, 50000)
    ax.set_ylim(60, 100)
    ax.set_xscale('log')
    ax.grid(True)
    ax.set_xlabel('Training set size')
    ax.set_title('Accuracy (%) on the validation data')
    plt.show()

if __name__ == "__main__":
    main()

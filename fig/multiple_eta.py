"""multiple_eta
~~~~~~~~~~~~~~~

This program shows how different values for the learning rate affect
training.  In particular, we'll plot out how the cost changes using
three different values for eta.

"""

# Standard library
import json
import random
import sys

# My library
sys.path.append('../src/')
import mnist_loader
import network2

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np

# Constants
LEARNING_RATES = [0.025, 0.25, 2.5]
COLORS = ['#2A6EA6', '#FFCD33', '#FF7033']
NUM_EPOCHS = 30

def main():
    run_networks()
    make_plot()

def run_networks():
    """Train networks using three different values for the learning rate,
    and store the cost curves in the file ``multiple_eta.json``, where
    they can later be used by ``make_plot``.

    """
    # Make results more easily reproducible
    random.seed(12345678)
    np.random.seed(12345678)
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    results = []
    for eta in LEARNING_RATES:
        print "\nTrain a network using eta = "+str(eta)
        net = network2.Network([784, 30, 10])
        results.append(
            net.SGD(training_data, NUM_EPOCHS, 10, eta, lmbda=5.0,
                    evaluation_data=validation_data, 
                    monitor_training_cost=True))
    f = open("multiple_eta.json", "w")
    json.dump(results, f)
    f.close()

def make_plot():
    f = open("multiple_eta.json", "r")
    results = json.load(f)
    f.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for eta, result, color in zip(LEARNING_RATES, results, COLORS):
        _, _, training_cost, _ = result
        ax.plot(np.arange(NUM_EPOCHS), training_cost, "o-",
                label="$\eta$ = "+str(eta),
                color=color)
    ax.set_xlim([0, NUM_EPOCHS])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cost')
    plt.legend(loc='upper right')
    plt.show()

if __name__ == "__main__":
    main()

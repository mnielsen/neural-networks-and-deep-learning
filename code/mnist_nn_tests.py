"""
mnist_nn_tests
~~~~~~~~~~~~~~

A few simple tests for the module mnist_nn.py."""

#### Libraries
# My library
from mnist_nn import Network

# Third-party libraries
import numpy as np


def main():
    test_feedforward()
    test_SGD()

def test_feedforward():
    """ Test the Network.feedforward method.  We do this by setting up
    a 3-layer network to compute the XOR function, and verifying that
    the outputs are as they should be."""
    print "\n\nTesting a neural network to compute the XOR function"
    net = Network([2, 2, 1])
    net.biases = [np.array([[-10.0], [10.0]]),
                  np.array([[10.0]])]
    net.weights = [np.array([[20.0, -20.0], [20.0, -20.0]]),
                   np.array([[20.0, -20.0]])]
    evaluate(net, test_harness_training_data())
    return net

def test_SGD():
    print "\n\nTesting stochastic gradient descent to find a network for XOR."
    net = Network([2, 2, 1])
    training_data = test_harness_training_data()
    net.SGD(training_data, 10000, 4, 0.1, 0.0001)
    evaluate(net, training_data)
    return net

def evaluate(net, test_data):
    """Evaluate ``net`` against the ``test_data``, comparing actual outputs
    to desired outputs."""
    failure = False # flag to indicate whether any tests have failed
    for x, y in test_data:
        output = net.feedforward(x)
        print "\nInput:\n%s" % x
        print "Expected output: {:.3f}".format(float(y))
        print "Actual output: {:.3f}".format(float(output))
        if abs(output - y) < 0.2:
            print "Test passed"
        else:
            print "Test failed"
            failure = True
    print "\nOne or more tests failed" if failure else "\nAll tests passed"
    
def test_harness_training_data():
    "Return a test harness containing training data for XOR."
    return [
        (np.array([[0.0], [0.0]]), np.array([[0.0]])),
        (np.array([[0.0], [1.0]]), np.array([[1.0]])),
        (np.array([[1.0], [0.0]]), np.array([[1.0]])),
        (np.array([[1.0], [1.0]]), np.array([[0.0]]))]

if __name__ == "__main__":
    main()

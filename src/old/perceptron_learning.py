"""
perceptron_learning
~~~~~~~~~~~~~~~~~~~

Demonstrates how a perceptron can learn the NAND gate, using the
perceptron learning algorithm."""

#### Libraries
# Third-party library
import numpy as np

class Perceptron(object):
    """ A Perceptron instance can take a function and attempt to
    ``learn`` a bias and set of weights that compute that function,
    using the perceptron learning algorithm."""

    def __init__(self, num_inputs=2):
        """ Initialize the perceptron with the bias and all weights
        set to 0.0. ``num_inputs`` is the number of input bits to the
        perceptron."""
        self.num_inputs = num_inputs
        self.bias = 0.0
        self.weights = np.zeros(num_inputs)
        # self.inputs is a convenience attribute.  It's a list containing
        # all possible binary inputs to the perceptron.  E.g., for three
        # inputs it is: [np.array([0, 0, 0]), np.array([0, 0, 1]), ...]
        self.inputs = [np.array([int(y)
                        for y in bin(x).lstrip("0b").zfill(num_inputs)])
                       for x in xrange(2**num_inputs)]

    def output(self, x):
        """ Return the output (0 or 1) from the perceptron, with input
        ``x``."""
        return 1 if np.inner(self.weights, x)+self.bias > 0 else 0

    def learn(self, f, eta=0.1):
        """ Find a bias and a set of weights for a perceptron that
        computes the function ``f``. ``eta`` is the learning rate, and
        should be a small positive number.  Does not terminate when
        the function cannot be computed using a perceptron."""
        # initialize the bias and weights with random values
        self.bias = np.random.normal()
        self.weights = np.random.randn(self.num_inputs)
        number_of_errors = -1
        while number_of_errors != 0:
            number_of_errors = 0
            print "Beginning iteration"
            print "Bias: {:.3f}".format(self.bias)
            print "Weights:", ", ".join(
                "{:.3f}".format(wt) for wt in self.weights)
            for x in self.inputs:
                error = f(x)-self.output(x)
                if error:
                    number_of_errors += 1
                    self.bias = self.bias+eta*error
                    self.weights = self.weights+eta*error*x
            print "Number of errors:", number_of_errors, "\n"

def f(x):
    """ Target function for the perceptron learning algorithm.  I've
    chosen the NAND gate, but any function is okay, with the caveat
    that the algorithm won't terminate if ``f`` cannot be computed by
    a perceptron."""
    return int(not (x[0] and x[1]))

if __name__ == "__main__":
    Perceptron(2).learn(f, 0.1)

"""
mnist_nn
~~~~~~~~

A classifier program which trains a neural network using data from the
MNIST training set of handwritten digits, and then evaluates
performance using the MNIST test data set.  In general, the code
should be easily adaptable to other purposes."""

#### Libraries
# Standard library
import random

# My libraries
import mnist_loader # to load the MNIST data

# Third-party libraries
import numpy as np

#### Program parameters
NETWORK = [784, 100, 30, 10] # number of neurons in each layer
EPOCHS = 30
MINI_BATCH_SIZE = 10
ETA = 0.01
LMBDA = 0.001

#### Main program
def main():
    training_data, test_inputs, actual_test_results = load_data()
    net = Network(NETWORK)
    net.SGD(training_data, EPOCHS, MINI_BATCH_SIZE, ETA, LMBDA,
            test=True, test_inputs=test_inputs, 
            actual_test_results=actual_test_results)

class Network():

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        layers of a feedforward network.  For example, if the list was
        [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        omit to set any biases for those neurons, since biases are
        only ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        "Return the output of the network if `a` is input."
        for b, w in zip(self.biases, self.weights):
            a = sigmoid_vec(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda, test=False, test_inputs=None, actual_test_results=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  Set ``test`` to ``True`` to evaluate the
        network against the test training data after each epoch, and
        to print out partial progress.  This is useful for tracking
        partial progress, but slows things down quite a bit.  If
        ``test`` is set, then appropriate ``test_inputs`` and
        ``actual_test_results`` must also be supplied.
        """
        if test: n = len(test_inputs)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                self.backprop(mini_batch, eta=eta, lmbda=lmbda)
            if test:
                print "Epoch {}: {} / {}".format(
                    j, self.evaluate(test_inputs, actual_test_results), n)

    def backprop(self, training_data, eta, lmbda, gradient_checking=False):
        """Update the network's weights and biases by applying a
        single iteration of gradient descent using backpropagation.
        The ``training_data`` is a list of tuples ``(x, y)``.  The
        other non-optional parameters are self-explanatory.  The
        optional flag ``gradient_checking`` determines whether or not
        gradient checking is done."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in training_data:
            # feedforward
            activation = x
            activations = [x] # list to store all the activations
            zs = [] # list to store all the z vectors
            for b, w in zip(self.biases, self.weights):
                z = np.dot(w, activation)+b
                zs.append(z)
                activation = sigmoid_vec(z)
                activations.append(activation)
            # backward pass
            delta = cost_derivative(activations[-1], y) * \
                sigmoid_prime_vec(zs[-1])
            nabla_b[-1] += delta
            nabla_w[-1] += np.dot(delta, np.transpose(activations[-2]))
            if gradient_checking:
                self._gradient_check(1, activations[-2], delta, x, y)
            # Note that the variable l in the loop below is used a
            # little differently to the book.  Here, l = 1 means the
            # last layer of neurons, l = 2 is the second-last layer,
            # and so on.  It's a renumbering of the scheme used in the
            # book, used to take advantage of the fact that Python can
            # use negative indices in lists.
            for l in xrange(2, self.num_layers):
                z = zs[-l]
                spv = sigmoid_prime_vec(z)
                delta = np.dot(np.transpose(self.weights[-l+1]), delta) * spv
                nabla_b[-l] += delta
                nabla_w[-l] += np.dot(delta, np.transpose(activations[-l-1]))
                if gradient_checking:
                    self._gradient_check(l, activations[-l-1], delta, x, y)
        # Add the regularization terms to the gradient for the weights
        nabla_w = [nw+lmbda*w for nw, w in zip(nabla_w, self.weights)]
        self.weights = [w-eta*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-eta*nb for b, nb in zip(self.biases, nabla_b)]

    def _gradient_check(self, l, activations, delta, x, y):
        """ Do a gradient check for the ``l``th layer from the end,
        which we'll dub the ``-l``th layer.  ``activations`` is for
        the ``-l-1`` layer, and ``delta`` is for the ``-l`` layer.
        The input to the network is ``x``, and the desired output is
        ``y``."""
        print "\nGradient check for the -%s layer" % l
        backprop_nabla_b = delta
        backprop_nabla_w = np.dot(delta, np.transpose(activations))
        d = 0.00001 # Delta for the biases and weights for gradient check
        # Gradient check the biases
        gradient_check_nabla_b = np.zeros((self.sizes[-l], 1))
        for j in xrange(self.sizes[-l]):
            net1 = self.copy()
            net1.biases[-l][(j,0)] += d
            net2 = self.copy()
            net2.biases[-l][(j,0)] -= d
            gradient_check_nabla_b[j] = (net1.cost(x, y)-net2.cost(x, y))/(2*d)
        print "Squared Euclidean error in the gradient for biases: %s" % \
            np.sum((backprop_nabla_b-gradient_check_nabla_b)**2)
        # Gradient check the weights
        gradient_check_nabla_w = np.zeros((self.sizes[-l], self.sizes[-l-1]))
        for j in xrange(self.sizes[-l]):
            for k in xrange(self.sizes[-l-1]):
                net1 = self.copy()
                net1.weights[-l][(j,k)] += d
                net2 = self.copy()
                net2.weights[-l][(j,k)] -= d
                gradient_check_nabla_w[j, k] = \
                    (net1.cost(x, y)-net2.cost(x, y))/(2*d)
        print "Squared Euclidean error in the gradient for weights: %s" % \
            np.sum((backprop_nabla_w-gradient_check_nabla_w)**2)

    def copy(self):
        "Return a copy of ``self``, with the same biases and weights."
        net = Network(self.sizes)
        net.biases = [np.copy(b) for b in self.biases]
        net.weights = [np.copy(w) for w in self.weights]
        return net

    def evaluate(self, test_inputs, actual_test_results):
        test_results = [np.argmax(self.feedforward(x)) for x in test_inputs]
        return sum(int(x == y) 
                   for x, y in zip(test_results, actual_test_results))
        
    def cost(self, x, y):
        return np.sum((self.feedforward(x)-y)**2 / 2.0)

    def cost_derivative(output_activations, y):
        return (output_activations-y) 

    def total_cost(self, training_data, regularization=0.001):
        training_cost = sum(self.cost(x, y) for x, y in training_data)
        regularization_cost = regularization * sum(
            np.sum(w**2) for w in self.weights)/2.0
        return training_cost+regularization_cost

#### Miscellaneous functions
def load_data():
    """Return a tuple containing ``(training_data, test_inputs,
    actual_test_results)`` from the MNIST data.  Makes use of
    ``mnist_loader.load_data()``, but does some additional processing
    to put the data in a useful format.  Consult the ``mnist_loader``
    code for more on the format used by that module.

    ``training_data`` is a list containing 50,000 2-tuples ``(x, y)``.
    ``x`` is a 784-dimensional numpy.ndarray containing the input
    image.  ``y`` is a 10-dimensional numpy.ndarray representing the
    unit vector corresponding to the correct digit for ``x``.

    ``test_inputs`` is a list containing 10,000 x 784-dimensional
    numpy.ndarray objects, representing test images.

    ``actual_test_results`` is a list containing the 10,000 digit
    values (integers) corresponding to the ``test_inputs``. 

    Obviously, we're using slightly different formats for the training
    and test data.  These formats turn out to be the most convenient
    for use elsewhere in the code."""
    training_data, validation_data, test_data = mnist_loader.load_data()
    inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]
    results = [vectorized_result(y) for y in training_data[1]]
    training_data = zip(inputs, results)
    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    return (training_data, test_inputs, test_data[1])

def sigmoid(z):
    """The sigmoid function.  Note that it checks to see whether ``z``
    is very negative, to avoid overflow errors in the exponential
    function.  No corresponding test of being very positive is
    necessary --- ordinary Python arithmetic will deal just fine with
    that case."""
    if z < -700:
        return 0.0
    else:
        return 1.0/(1.0+np.exp(-z))

sigmoid_vec = np.vectorize(sigmoid)

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

sigmoid_prime_vec = np.vectorize(sigmoid_prime)

def vectorized_result(j):
    """ Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is a convenience function
    which is used to convert XXX."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


#### Testing

def test_feedforward():
    """ Test the Network.feedforward method.  We do this by setting up
    a 3-layer network to compute the XOR function, and verifying that
    the outputs are as they should be."""
    net = Network([2, 2, 1])
    net.biases = [np.array([[-10.0], [10.0]]),
                  np.array([[10.0]])]
    net.weights = [np.array([[20.0, -20.0], [20.0, -20.0]]),
                   np.array([[20.0, -20.0]])]
    failure = False # flag to indicate whether any tests have failed
    print "Testing a neural network to compute the XOR function"
    for x, y in test_harness_training_data():
        output = net.feedforward(x)
        print "\nInput:\n%s" % x
        print "Expected output: {0:.3f}".format(float(y))
        print "Actual output: {0:.3f}".format(float(output))
        if abs(output - y) < 0.001:
            print "Test passed"
        else:
            print "Test failed"
            failure = True
    print "\nOne or more tests failed" if failure else "\nAll tests passed"

def test_backprop(n, gradient_checking=False):
    net = Network([2, 2, 1])
    training_data = test_harness_training_data()
    for j in xrange(n):
        net.backprop(test_harness_training_data(), eta=0.1, 
                     regularization=0.0001, gradient_checking=gradient_checking)
        print net.total_cost(training_data, 0.0001)
    return net
    
def test_harness_training_data():
    "Return a test harness containing training data for XOR."
    return [
        (np.array([[0.0], [0.0]]), np.array([[0.0]])),
        (np.array([[0.0], [1.0]]), np.array([[1.0]])),
        (np.array([[1.0], [0.0]]), np.array([[1.0]])),
        (np.array([[1.0], [1.0]]), np.array([[0.0]]))]

if __name__ == "__main__":
    main()

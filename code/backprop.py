"""
backprop
~~~~~~~~

Uses backpropagation and stochastic gradient descent to implement a
handwritten digit recognizer."""

#### Libraries
# Standard library
from collections import defaultdict
import cPickle

# Third-party libraries
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

#### Baseline: Average darkness classifier
####
#### Classifies images by computing how dark the image is, and then
#### returning whichever digit had the closest average darkness in the
#### training data

def avg_darkness(training_data):
    """ Return a defaultdict whose keys are the digits, 0 through 9.
    For each digit we compute the average darkness of images
    containing that digit.  The darkness for any particular image is
    just the sum of the darknesses for each pixel."""
    digit_counts = defaultdict(int)
    darkness = defaultdict(float)
    avgs = defaultdict(float)
    for image, digit in zip(training_data[0], training_data[1]):
        digit_counts[digit] += 1
        darkness[digit] += sum(image)
    for digit, n in digit_counts.iteritems():
        avgs[digit] = darkness[digit] / n
    return avgs

def guess_digit(image, avg_darkness):
    darkness = sum(image)
    distance = {k: abs(v-darkness) for k, v in avg_darkness.iteritems()}
    return min(distance, key=distance.get)

def test_average_darkness_baseline():
    training_data, validation_data, test_data = load_data()
    avgs = avg_darkness(training_data)
    correct = sum(int(guess_digit(image, avgs) == digit)
                  for image, digit in zip(test_data[0], test_data[1]))
    print "Baseline classifier using average darkness of image."
    print "%s of %s values correct." % (correct, len(test_data[1]))

#### Baseline: SVM classifier
def test_svm_baseline():
    """
    Use an SVM to classify MNIST digits.  Print the number which are
    classified correctly, and draw a figure showing the first ten
    images which are misclassified."""
    training_data, validation_data, test_data = load_data()
    clf = svm.SVC()
    clf.fit(training_data[0], training_data[1])
    predictions = [int(v) for v in clf.predict(test_data[0])]
    num_correct = sum(int(x == y) for x, y in zip(predictions, test_data[1]))
    print "Baseline classifier using an SVM."
    print "%s of %s values correct." % (num_correct, len(test_data[1]))
    # indices of the images where we fail
    failures = [j for (j, z) in enumerate(zip(predictions, test_data[1]))
                if z[0] != z[1]]
    # the first ten images where we fail
    images = [test_data[0][failures[j]] for j in xrange(10)]
    fig = plt.figure()
    for j in xrange(1, 11):
        ax = fig.add_subplot(1, 10, j)
        ax.matshow(images[j-1], cmap = matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.show()

#### Neural network classifier

class Network():

    def __init__(self, sizes):
        """The list `sizes` contains the number of neurons in the
        layers of a feedforward network.  For example, if the list was
        [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1."""
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

    def backprop(self, training_data, eta=0.1, 
                 regularization=0.01, testing=False):
        """Update the network's weights and biases by applying a
        single iteration of gradient descent using backpropagation.
        The ``training_data`` is a list of tuples ``(x, y)`` and
        ``eta`` is the learning rate.  The variable ``regularization``
        is the value of the regularization paremeter.  The flag
        ``testing`` determines whether or not gradient checking is
        done."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in training_data:
            # feedforward stage
            activation = x
            activations = [x] # list to store all the activations
            zs = [] # list to store all the z vectors
            for b, w in zip(self.biases, self.weights):
                z = np.dot(w, activation)+b
                zs.append(z)
                activation = sigmoid_vec(z)
                activations.append(activation)
            # backward pass
            delta = (activations[-1]-y) * sigmoid_prime_vec(zs[-1])
            nabla_b[-1] += delta
            nabla_w[-1] += np.dot(delta, np.transpose(activations[-2]))
            for l in xrange(2, self.num_layers):
                z = zs[-l]
                spv = sigmoid_prime_vec(z)
                delta = np.dot(
                    np.transpose(self.weights[-l+1]), delta) * spv
                nabla_b[-l] += delta
                nabla_w[-l] += np.dot(delta, np.transpose(activations[-l-1]))
        nabla_w = [nw+regularization*w for nw, w in zip(nabla_w, self.weights)]
        self.weights = [w-eta*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-eta*nb for b, nb in zip(self.biases, nabla_b)]
        if testing:
            pass
            #print "\nBackprop: %s" % delta_nabla[0][(1, 0)]
            #print "Numerical: %s" % self.comparison_gradient(
            #    activation, y, 0, 0, 1)

    def cost(self, x, y):
        return (self.feedforward(x)-y)**2 / 2.0

    def comparison_gradient(self, activation, y, j, k, l):
        """
        Return the partial derivative of the cost function for
        ``activation`` with respect to the weight joining the k'th and
        l'th neurons in the j'th layer of weights.  The input ``y`` is
        the correct output value for the network.  This partial
        derivative is computed numerically, not using backpropagation.
        It's included as a test comparison to be used against the
        computation done using backpropagation."""
        # Construct two networks with the appropriate weight modified
        delta = 0.00001 # amount to vary the weight by
        net1 = Network(self.sizes)
        net1.biases = [np.copy(bias) for bias in self.biases]
        net1.weights = [np.copy(wt) for wt in self.weights]
        net1.weights[j][(l, k)] += delta
        net2 = Network(self.sizes)
        net2.biases = [np.copy(bias) for bias in self.biases]
        net2.weights = [np.copy(wt) for wt in self.weights]
        net2.weights[j][(l, k)] -= delta
        return (net1.cost(activation, y)-net2.cost(activation, y))/(2*delta)

    def error(self, training_data, regularization=0.01):
        training_error = sum(euclidean_error(self.feedforward(x)-y) 
                   for x, y in training_data)
        regularization_error = regularization * sum(
            np.sum(wt*wt) for wt in self.weights)/2.0
        return training_error+regularization_error

#### Miscellaneous functions
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

sigmoid_vec = np.vectorize(sigmoid)

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

sigmoid_prime_vec = np.vectorize(sigmoid_prime)

def euclidean_error(delta):
    return np.linalg.norm(delta)**2/2.0

def vectorized_result(j):
    """ Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is a convenience function
    which is used to convert XXX."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

#### Neural network MNIST classifier
def neural_network_classifier():
    training_data, validation_data, test_data = load_data()
    inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]
    results = [vectorized_result(y) for y in training_data[1]]
    training_data = zip(inputs, results)
    net = Network([784, 10])
    for j in xrange(10):
        print net.error(training_data, regularization=0.001)
        net.backprop(training_data, eta=0.1, regularization=0.001)

#### Testing

def test_backprop(n):
    net = Network([2, 2, 1])
    training_data = test_harness_training_data()
    for j in xrange(n):
        net.backprop(test_harness_training_data(), eta=0.1, 
                     regularization=0.0001)
        error = sum((net.feedforward(x)-y)**2/2 for x, y in training_data)
        print net.error(training_data, 0.0001)
    return net

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
    
def test_harness_training_data():
    "Return a test harness containing training data for XOR."
    return [
        (np.array([[0.0], [0.0]]), np.array([[0.0]])),
        (np.array([[0.0], [1.0]]), np.array([[1.0]])),
        (np.array([[1.0], [0.0]]), np.array([[1.0]])),
        (np.array([[1.0], [1.0]]), np.array([[0.0]]))]


#### Miscellanea
def load_data():
    """ Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data."""
    # TODO: Ask Ng if I can host it (where?)
    f = open('mnist.pkl', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

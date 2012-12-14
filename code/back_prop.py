"""
back_prop
~~~~~~~~~

Uses back-propagation and stochastic gradient descent to implement a
basic handwritten digit recognizer."""


#### Libraries
# Standard library
from collections import defaultdict
import cPickle

# Third-party libraries
import matplotlib.pyplot as plt
import numpy
from sklearn import svm

def sigmoid(z):
    return 1.0/(1.0+numpy.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

class Network():

    def __init__(layers, sizes):
        self.layers = layers
        self.sizes = size
        #self.weights = [numpy matrix for j in xrange(self.layers)]

def stochastic_gradient_descent():
    pass
    #initialize  the weights randomly.
    #for iterations:
    #    randomly pick out a mini-batch of training examples and their values
        

#### Baseline: Average darkness classifier
####
#### Guesses which digit an image is by computing how dark the image
#### is, and then returns whichever digit had the closest average
#### darkness in the training data

def avg_darkness():
    """ Return a defaultdict whose keys are the digits, 0 through 9.
    For each digit we compute the average darkness of images
    containing that digit.  The darkness for any particular image is
    just the sum of the darknesses for each pixel."""
    digit_counts = defaultdict(int)
    darkness = defaultdict(float)
    avgs = defaultdict(float)
    for image, digit in zip(training_set[0], training_set[1]):
        digit_counts[digit] += 1
        darkness[digit] += sum(image)
    for digit, v in digit_counts.iteritems():
        avgs[digit] = darkness[digit] / v
    return avgs

def guess_digit(image, avg_darkness):
    darkness = sum(image)
    distance = {k: abs(v-darkness) for k, v in avg_darkness.iteritems()}
    return min(distance, key=distance.get)

def test_average_darkness_baseline():
    training_set, validation_set, test_set = load_data()
    avgs = avg_darkness()
    correct = sum(int(guess_digit(image, avgs) == digit)
                  for image, digit in zip(test_set[0], test_set[1]))
    print "Baseline classifier using average darkness of image."
    print "%s of %s values correct." % (correct, len(test_set[1]))

#### Baseline: SVM classifier
def test_svm_baseline():
    training_set, validation_set, test_set = load_data()
    clf = svm.SVC()
    clf.fit(training_set[0], training_set[1])
    predictions = [int(v) for v in clf.predict(validation_set[0])]
    correct = sum(int(x == y) for x, y in zip(predictions, validation_set[1]))
    print "Baseline classifier using an SVM."
    print "%s of %s values correct." % (correct, len(validation_set[1]))

#### Miscellanea
def load_data():
    """ Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data."""
    # TODO: Ask Ng if I can host it (where?)
    f = open('mnist.pkl', 'rb')
    training_set, validation_set, test_set = cPickle.load(f)
    f.close()
    return (training_set, validation_set, test_set)



#test_average_darkness_baseline()
test_svm_baseline()



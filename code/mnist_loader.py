"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc string for ``load_data``.
The library also contains a helper method ``load_data_nn`` which
returns the data in a format well adapted for use with our neural
network code.

Note that the code requires the file ``mnist.pkl``.  This is not
included in the repository.  It may be downloaded from:

http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
"""

#### Libraries
# Standard library
import cPickle

# Third-party libraries
import numpy as np


def load_data():
    """ Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    Note that the format the data is returned in is well adapted for
    use by scikit-learn's SVM method, but not so well adapted for our
    neural network code.  For that, see the wrapper function
    ``load_data_nn``.
    """
    f = open('mnist.pkl', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

def load_data_nn():
    """Return a tuple containing ``(training_data, test_inputs,
    actual_test_results)`` from the MNIST data.  The tuples are in a
    format optimized for use by our neural network code.  This
    function makesuse of ``load_data()``, but does some additional
    processing to put the data in the right format.  

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
    for use in our neural network code."""
    training_data, validation_data, test_data = load_data()
    inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]
    results = [vectorized_result(y) for y in training_data[1]]
    training_data = zip(inputs, results)
    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    return (training_data, test_inputs, test_data[1])

def vectorized_result(j):
    """ Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

"""
backprop2
~~~~~~~~~

This is a minor variation on the backprop module.  The main changes
are: (1) It uses the cross-entropy cost by default, instead of the
quadratic cost; (2) The weights and bias for each neuron are
initialized as a Gaussian random variable whose mean is zero and
standard deviation is one over the square-root of the neuron's fan-in
(instead of having mean zero and standard deviation one); and (3) The
feedforward method has been generalized to make it easier to propagate
activations through part of the network.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

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
        self.biases = [np.random.randn(y, 1)/np.sqrt(y) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(y) 
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a, start=0, end=None):
        """Return the result from feeding forward the activation ``a``
        from layer ``start`` through to layer ``end``.  Note that if
        ``end`` is ``None`` then this is interpreted as ``end =
        self.num_layers-1``, i.e., the default behaviour is to propagate
        through to the end of the network."""
        if end == None: end = self.num_layers-1
        for b, w in zip(self.biases, self.weights)[start:end]:
            a = sigmoid_vec(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda, test=False, test_inputs=None, actual_test_results=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  Set ``test`` to ``True`` to evaluate the
        network against the test data after each epoch, and to print
        out partial progress.  This is useful for tracking progress,
        but slows things down substantially.  If ``test`` is set, then
        appropriate ``test_inputs`` and ``actual_test_results`` must
        also be supplied.
        """
        if test: n = len(test_inputs)
        T = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                self.backprop(mini_batch, T, eta, lmbda, 
                              self.cost_cross_entropy,
                              self.cost_derivative_cross_entropy)
            if test:
                print "Epoch {}: {} / {}".format(
                    j, self.evaluate(test_inputs, actual_test_results), n)
            else:
                print "Epoch %s: %s" % (
                    j, 
                    sum(self.cost_cross_entropy(x, y) 
                        for (x, y) in training_data)) 

    def backprop(self, training_data, T, eta, lmbda, 
                 cost, cost_derivative):
        """Update the network's weights and biases by applying a
        single iteration of gradient descent using backpropagation.
        The ``training_data`` is a list of tuples ``(x, y)``.  It need
        not include the entire training data set --- it might be a
        mini-batch, or even a single training example.  ``T`` is the
        size of the total training set (which may not be the same as
        the size of ``training_data``).  The parameters ``eta`` and
        ``lmbda`` are the learning rate and regularization
        parameter. The two final parameters are functions used to
        compute the cost and cost derivative.  Examples of those
        functions are ``self.cost_cross_entropy`` and
        ``self.cost_derivative_cross_entropy``, as defined below."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        B = len(training_data)
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
            nabla_w[-1] += np.dot(delta, activations[-2].transpose())
            # Note that the variable l in the loop below is used a
            # little differently to the book.  Here, l = 1 means the
            # last layer of neurons, l = 2 is the second-last layer,
            # and so on.  It's a renumbering of the scheme used in the
            # book, used to take advantage of the fact that Python can
            # use negative indices in lists.
            for l in xrange(2, self.num_layers):
                z = zs[-l]
                spv = sigmoid_prime_vec(z)
                delta = np.dot(self.weights[-l+1].transpose(), delta) * spv
                nabla_b[-l] += delta
                nabla_w[-l] += np.dot(delta, activations[-l-1].transpose())
        # Add the regularization terms to the gradient for the weights
        nabla_w = [nw+(lmbda*B/T)*w for nw, w in zip(nabla_w, self.weights)]
        self.weights = [w-eta*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-eta*nb for b, nb in zip(self.biases, nabla_b)]

    def evaluate(self, test_inputs, actual_test_results):
        """Return the number of ``test_inputs`` for which the neural
        network outputs the correct result, i.e., the same result as
        given in ``actual_test_results``.  Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [np.argmax(self.feedforward(x)) for x in test_inputs]
        return sum(int(x == y) 
                   for x, y in zip(test_results, actual_test_results))
        
    def cost_quadratic(self, x, y):
        """Return the unregularized quadratic cost associated to the
        network, with input ``x`` and desired output ``y``."""
        return np.sum((self.feedforward(x)-y)**2)/2.0

    def cost_cross_entropy(self, x, y):
        """Return the unregularized cross entropy cost associated to
        the network, with input ``x`` and desired output ``y``."""
        a = self.feedforward(x)
        return np.sum(-y*np.log(a)-(1-y)*np.log(1-a)) 

    def cost_derivative_quadratic(self, a, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations, ``a``, and for an
        unregularized quadratic cost function."""
        return (a-y) 

    def cost_derivative_cross_entropy(self, a, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations, ``a``, and for the
        unregularized cross-entropy cost function."""
        return (a-y)/(a*(1-a)) 

    def evaluate_training_results(self, training_data):
        """Return the number of elements of the ``training_data`` that
        are correctly classified."""
        training_results = [np.argmax(self.feedforward(x[0])) for x in 
                            training_data]
        actual_training_results = [np.argmax(x[1]) for x in training_data]
        return sum(int(x == y) 
                   for x, y in zip(training_results, actual_training_results))


#### Miscellaneous functions
def minimal_cross_entropy(training_data):
    """
    Return the minimal output cross entropy for training data
    consisting of a list of values ``(x, y)``.  This is just
    -sum (y ln(y) + (1-y) ln(1-y))."""
    def rescale(y):
        """
        Rescale the vectors very slightly to ensure there are no
        problems with taking logarithms of 0.  This might cause a
        problem if there are a huge number of features (10**7 & up),
        in which case this code will need to be modified."""
        return 10**(-9)+(1-2*10**(-9))*y
    rescaled_training_data = [rescale(y) for (_, y) in training_data]
    return np.sum(-np.sum(y*np.log(y)+(1-y)*np.log(1-y))
                for y in rescaled_training_data)

def sigmoid(z):
    """The sigmoid function.  Note that it checks to see whether ``z``
    is very negative, to avoid overflow errors in the exponential
    function.  No corresponding test of ``z`` being very positive is
    necessary --- ordinary Python arithmetic deals just fine with that
    case."""
    return 0.0 if z < -700 else 1.0/(1.0+np.exp(-z))

sigmoid_vec = np.vectorize(sigmoid)

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

sigmoid_prime_vec = np.vectorize(sigmoid_prime)

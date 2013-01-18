"""
gradient_descent_hack
~~~~~~~~~~~~~~~~~~~~~

This program uses gradient descent to learn weights and biases for a
three-neuron network to compute the XOR function.  The program is a
quick-and-dirty hack meant to illustrate the basic ideas of gradient
descent, not a cleanly-designed and generalizable implementation."""

#### Libraries
# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def neuron(w, x):
    """ Return the output from the sigmoid neuron with weights ``w``
    and inputs ``x``.  Both are numpy arrays, with three and two
    elements, respectively.  The first input weight is the bias."""
    return sigmoid(w[0]+np.inner(w[1:], x))

def h(w, x):
    """ Return the output from the three-neuron network with weights
    ``w`` and inputs ``x``.  Note that ``w`` is a numpy array with
    nine elements, consisting of three weights for each neuron (the
    bias plus two input weights).  ``x`` is a numpy array with just
    two elements."""
    neuron1_out = neuron(w[0:3], x) # top left neuron
    neuron2_out = neuron(w[3:6], x) # bottom left neuron
    return neuron(w[6:9], np.array([neuron1_out, neuron2_out]))

# inputs and corresponding outputs for the function we're computing (XOR)
INPUTS = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]] 
OUTPUTS = [0.0, 1.0, 1.0, 0.0]

def cost(w):
    """ Return the cost when the neural network has weights ``w``.
    The cost is computed with respect to the XOR function."""
    return 0.5 * sum((y-h(w, np.array(x)))**2 for x, y in zip(INPUTS, OUTPUTS))

def partial(f, k, w):
    """ Return the partial derivative of the function ``f`` with
    respect to the ``k``th variable, at location ``w``.  Note that
    ``f`` must take a numpy array as input, and the partial derivative
    is evaluated with respect to the ``k``th element in that array.
    Similarly, ``w`` is a numpy array which can be used as input to
    ``f``."""
    w_plus, w_minus = w.copy(), w.copy()
    w_plus[k] += 0.01 # using epsilon = 0.01
    w_minus[k] += -0.01
    return (f(w_plus)-f(w_minus))/0.02
    
def gradient_descent(cost, eta, n):
    """ Perform ``n`` iterations of the gradient descent algorithm to
    minimize the ``cost`` function, with a learning rate ``eta``.
    Return a tuple whose first entry is an array containing the final
    weights, and whose second entry is a list of the values the
    ``cost`` function took at different iterations."""
    w = np.random.uniform(-1, 1, 9) # initialize weights randomly
    costs = []
    for j in xrange(n):
        c = cost(w)
        print "Current cost: {0:.3f}".format(c)
        costs.append(c)
        gradient = [partial(cost, k, w) for k in xrange(9)]
        w = np.array([wt-eta*d for wt, d in zip(w, gradient)])
    return w, costs

def main():
    """ Perform gradient descent to find weights for a sigmoid neural
    network to compute XOR.  10,000 iterations are used.  Outputs the
    final value of the cost function, the final weights, and plots a
    graph of cost as a function of iteration."""
    w, costs = gradient_descent(cost, 0.1, 10000)
    print "\nFinal cost: {0:.3f}".format(cost(w))
    print "\nFinal weights: %s" % w
    plt.plot(np.array(costs))
    plt.xlabel('iteration')
    plt.ylabel('cost')
    plt.title('How cost decreases with the number of iterations')
    plt.show()

if __name__ == "__main__":
    main()

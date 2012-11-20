"""
gradient_descent_hack
~~~~~~~~~~~~~~~~~~~~~

This program uses gradient descent to learn weights and biases for a
three-neuron network to compute the XOR function.  The program is a
quick-and-dirty hack to quickly illustrate the ideas."""

#### Libraries
# Third-party libraries
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy

def sigmoid(z):
    return 1.0/(1.0+numpy.exp(-z))

def neuron(w, x):
    """ Return the output from the sigmoid neuron with weights ``w``
    and inputs ``x``.  Assumes the first input weight is the bias."""
    return sigmoid(w[0]+numpy.inner(w[1:], x))

def h(w, x):
    """ Return the output from the three-neuron network with weights
    ``w`` and inputs ``x``.  Note that ``w`` is an array with 9
    elements, consisting of three weights for each neuron.  ``x`` is
    an array with just two elements."""
    neuron1_out = neuron(w[0:3], x)
    neuron2_out = neuron(w[3:6], x)
    return neuron(w[6:9], numpy.array([neuron1_out, neuron2_out]))

def cost(w):
    """ Return the cost when the neural network has weights ``w``.
    The cost is computed with respect to the XOR function."""
    # inputs converted below to arrays
    inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]] 
    outputs = [0.0, 1.0, 1.0, 0.0] # corresponding outputs for XOR
    return sum((y-h(w, numpy.array(x)))**2 for x, y in zip(inputs, outputs))

def partial(f, k, w):
    """ Return the partial derivative of the function ``f`` with
    respect to the ``k``th variable, at location ``w``.  Note that
    ``f`` must take a single numpy array as input, and the partial
    derivative is evaluated with respect to the ``k``th element in
    that array.  Similarly, ``w`` is a numpy array which can be used
    as input to ``f``."""
    w_plus, w_minus = w.copy(), w.copy()
    w_plus[k] += 0.01
    w_minus[k] += -0.01
    return (f(w_plus)-f(w_minus))/0.02
    
def gradient_descent(cost, d, eta, n):
    w = numpy.random.uniform(-1, 1, d) # initialize randomly
    costs = []
    for j in xrange(n):
        c = cost(w)
        print "Current cost: {0:.3f}".format(c)
        costs.append(c)
        gradient = [partial(cost, k, w) for k in xrange(d)]
        w = numpy.array([wt-eta*p for wt, p in zip(w, gradient)])
    return w, costs

def main():
    w, costs = gradient_descent(cost, 9, 0.1, 1000)
    print "\nFinal cost: {0:.3f}".format(cost(w))
    #print "Final weights: "+w
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("iteration", fontsize=16)
    ax.set_ylabel("cost", fontsize=16)
    ax.set_title("gradient descent", fontsize=20)
    # ax.set_xlim(XMIN, XMAX)
    # ax.set_ylim(YMIN, YMAX)
    ax.grid(True)
    x = numpy.arange(1000)
    ax.plot(x, numpy.array(costs), color="tomato")
    plt.show()
    #fig.savefig("FILENAME.png")

if __name__ == "__main__":
    main()

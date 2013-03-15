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


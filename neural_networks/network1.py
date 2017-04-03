import numpy as np
import random

## Sigmoid function
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))


class Network(object):

    """  Neural network class
    
    Attributes:
        sizes: List specifying number of neurons in each layer of the neural network 
                (Ex. [2 4 3] - network with 2 neurons in first (input) layer, 4 neurons in second layer, and 3 neurons in third layer)
        num_layers: Integer, number of layers in the network
        biases: List; ith element contains column-vector with biases for i+2 layer of neural network; 
                biases are omitted for input layer
        weights: List; ith element contains matrix of weights connecting i+1 and i+2 layers of the neural network              
                
        
    """

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]


    def feedforward(self, a):
        """ Calculates output of the neural network given input a
        
        :param a: numpy [n,1] vector, input to the neural network     
        :return: output of the neural network
        """

        for (b, w) in zip (self.biases, self.weights):
            a = sigmoid(w.dot(a) + b)

        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):
        """ Training neural network using stochastic gradient descent
        
        :param training_data: list of tuples (x, y), where x is training input and y is a label
        :param epochs: integer, number of SGD epochs
        :param mini_batch_size: integer, size of mini batch
        :param eta: float, learning rate of gradient descent mechanism
        :param test_data: list of tuples (x, y), where x is test input and y is a label
        :return: Updates weight and biases in the neural network
        """

        if test_data: n_test = len(test_data)
        n = len(training_data)

        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """
        Updating weights and biases in the neural network by applying backpropagation mechanism to a single mini batch
        :param mini_batch: list of tuples containing mini batch of a training data
        :param eta: learning rate
        :return: updates weights and biases
        """

        nabla_b = [np.zeros(b.shape()) for b in self.biases]
        nabla_w = [np.zeros(w.shape()) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]




"""
Got the code from https://github.com/MichalDanielDobrzanski/DeepLearningPython/pull/14/
"""

"""network3.py
~~~~~~~~~~~~~~
A Theano-based program for training and running simple neural
networks.
Supports several layer types (fully connected, convolutional, max
pooling, softmax), and activation functions (sigmoid, tanh, and
rectified linear units, with more easily added).
When run on a CPU, this program is much faster than network.py and
network2.py.  However, unlike network.py and network2.py it can also
be run on a GPU, which makes it faster still.
Because the code is based on Theano, the code is different in many
ways from network.py and network2.py.  However, where possible I have
tried to maintain consistency with the earlier programs.  In
particular, the API is similar to network2.py.  Note that I have
focused on making the code simple, easily readable, and easily
modifiable.  It is not optimized, and omits many desirable features.
This program incorporates ideas from the Theano documentation on
convolutional neural nets (notably,
http://deeplearning.net/tutorial/lenet.html ), from Misha Denil's
implementation of dropout (https://github.com/mdenil/dropout ), and
from Chris Olah (http://colah.github.io ).
"""

#### Libraries
# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np

import pytensor
import pytensor.link.jax
import pytensor.tensor as pt
import pytensor.tensor
from pytensor.tensor.math import sigmoid, tanh
from pytensor.tensor.special import softmax

# Activation functions for neurons
def linear(z): return z
# update with pt
def ReLU(z): return pt.maximum(0.0, z)

# 2d convolution
import jax
import jax.numpy as jnp
from jax.lax import conv_general_dilated, reduce_window

#### Constants
GPU = True
if GPU:
    print("Trying to run under a GPU.  If this is not desired, then modify "+\
        "network3.py\nto set the GPU flag to False.")

    # config has been replaced, instead we have cuda
    try: pytensor.config.device = 'cuda'
    except: pass # it's already set
    # recommended for GPU computation
    pytensor.config.floatX = 'float32'

    print(f"PyTensor is running on: {pytensor.config.device}")
    exit()
else:
    print("Running with a CPU. If this is not desired, then the modify "+\
        "network3.py to set\nthe GPU flag to True.")

#### Load the MNIST data
def load_data_shared(filename="../data/mnist.pkl.gz"):
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.
        """
        # shared is still the same between theano and pytensor
        shared_x = pytensor.shared(
            np.asarray(data[0], dtype=pytensor.config.floatX), borrow=True)
        
        # shared is still the same between theano and pytensor
        shared_y = pytensor.shared(
            np.asarray(data[1], dtype=pytensor.config.floatX), borrow=True)
        
        # update cast to pytensor.tensor (pt) instead of theano.tensor (T)
        return shared_x, pt.cast(shared_y, "int32")
    return [shared(training_data), shared(validation_data), shared(test_data)]

class ConvPoolLayer(object):
    """Used to create a combination of a convolutional and a max-pooling
    layer.  A more sophisticated implementation would separate the
    two, but for our purposes we'll always use them together, and it
    simplifies the code, so it makes sense to combine them.
    """

    def __init__(self, filter_shape, image_shape, poolsize=(2, 2),
                 activation_fn=sigmoid):
        """`filter_shape` is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and the
        filter width.
        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.
        `poolsize` is a tuple of length 2, whose entries are the y and
        x pooling sizes.
        """
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn=activation_fn

        # initialize weights and biases
        n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
        
        # theano.shared == pytensor.shared
        self.w = pytensor.shared(
            np.asarray(
                np.random.normal(loc=0, scale=np.sqrt(1.0/n_out), size=filter_shape),
                # theano.config.floatX == pytensor.config.floatX
                dtype=pytensor.config.floatX),
            borrow=True)
        
        # theano.shared == pytensor.shared
        self.b = pytensor.shared(
            np.asarray(
                np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                # theano.config.floatX == pytensor.config.floatX
                dtype=pytensor.config.floatX),
            borrow=True)
        
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        # this section is not possible using just pytensor
        # need to also use JAX for the 2d convolution 

        # Assume self.inpt, self.w, self.filter_shape, and self.image_shape are defined
        input_tensor = self.inpt  # Shape: (batch, channels, height, width)
        filters = self.w  # Shape: (out_channels, in_channels, filter_height, filter_width)

        # Stride (assumed to be 1x1 unless specified otherwise)
        stride = (1, 1)

        # Padding: Choose 'SAME' to keep the output size similar to the input
        padding = "SAME"  # Theano's default padding behavior is similar to "SAME"

        # Perform convolution
        conv_out = conv_general_dilated(
            lhs=input_tensor,  # Input tensor
            rhs=filters,  # Convolution filters
            window_strides=stride,  # Stride for convolution
            padding=padding  # Padding type
        )

        # UPGRADE: pytensor reshape
        self.inpt = pt.reshape(inpt, self.image_shape)
        
        # Pooling parameters (e.g., poolsize=(2, 2), ignore_border=True)
        window_shape = (2, 2)  # Pooling window size
        strides = (2, 2)  # Stride for pooling
        padding = 'VALID'  # Padding type (no padding at the borders)

        # Apply max pooling using jax.lax.reduce_window
        pooled_out = reduce_window(conv_out, -jnp.inf, jax.lax.max, window_shape, strides, padding)
        
        self.output = self.activation_fn(
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output_dropout = self.output # no dropout in the convolutional layers

class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        self.n_in = n_in                    # n input neurons
        self.n_out = n_out                  # n output neurons
        self.activation_fn = activation_fn  # activation function
        self.p_dropout = p_dropout          # probability of dropping out (reduce overfitting)
        ### Initialize weights and biases
        # theano.shared == pytensor.shared
        self.w = pytensor.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)),
                # theano.config.floatX == pytensor.config.floatX
                dtype=pytensor.config.floatX),
            name='w', borrow=True)
        
        # theano.shared == pytensor.shared
        self.b = pytensor.shared(
            # theano.config.floatX == pytensor.config.floatX
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                       dtype=pytensor.config.floatX),
            name='b', borrow=True)
        
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        # UPGRADE: use pytensor reshape
        self.inpt = pt.reshape(inpt, (mini_batch_size, self.n_in))
        self.output = self.activation_fn(
            # UPGRADE: T.dot == pt.dot
            
            (1-self.p_dropout)*pt.dot(self.inpt, self.w) + self.b)
        # UPGRADE: T.argmax == pt.argmax
        self.y_out = pt.argmax(self.output, axis=1)

        # UPGRADE: use pytensor reshape
        inpt_dropout = pt.reshape(inpt_dropout, (mini_batch_size, self.n_in))

        self.inpt_dropout = dropout_layer(inpt_dropout, self.p_dropout)
        self.output_dropout = self.activation_fn(
            # T.dot == pt.dot
            pt.dot(self.inpt_dropout, self.w) + self.b)

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        # T.mean == pt.mean; T.eq == pt.eq
        return pt.mean(pt.eq(y, self.y_out))

class SoftmaxLayer(object):

    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases
        # theano.shared == pytensor.shared
        self.w = pytensor.shared(
            # theano.config.floatX == pytensor.config.floatX
            np.zeros((n_in, n_out), dtype=pytensor.config.floatX),
            name='w', borrow=True)
        # theano.shared == pytensor.shared
        self.b = pytensor.shared(
            # theano.config.floatX == pytensor.config.floatX
            np.zeros((n_out,), dtype=pytensor.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]
    
    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        # UPGRADE: use pytensor reshape
        self.inpt = pt.reshape(inpt, (mini_batch_size, self.n_in))
        # UPGRADE: T.dot == pt.dot
        self.output = softmax((1-self.p_dropout) * pt.dot(self.inpt, self.w) + self.b)
        # UPGRADE: T.argmax == pt.argmax
        self.y_out = pt.argmax(self.output, axis=1)
        # UPGRADE: use pytensor reshape
        inpt_dropout = pt.reshape(inpt_dropout, (mini_batch_size, self.n_in))

        self.inpt_dropout = dropout_layer(inpt_dropout, self.p_dropout)
        
        # UPGRADE: T.dot == pt.dot
        self.output_dropout = softmax(pt.dot(self.inpt_dropout, self.w) + self.b)

    def cost(self, net):
        "Return the log-likelihood cost."
        # T.mean == pt.mean; T.log == pt.log; T.arange == pt.arange
        return -pt.mean(pt.log(self.output_dropout)[pt.arange(net.y.shape[0]), net.y])

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        # T.mean == pt.mean; T.eq == pt.eq
        return pt.mean(pt.eq(y, self.y_out))

#### Main class used to construct and train networks
class Network(object):

    def __init__(self, layers: FullyConnectedLayer | ConvPoolLayer | SoftmaxLayer, mini_batch_size: int):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.
        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]

        # T.matrix == pt.matrix
        self.x = pt.matrix("x")
        # T.ivector == pt.ivector
        self.y = pt.ivector("y")
 
        init_layer = self.layers[0]

        # call the set_inpt for the respective layer provided as the first layer
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)

        for j in range(1, len(self.layers)): # xrange() was renamed to range() in Python 3.
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, lmbda=0.0):
        """Train the network using mini-batch stochastic gradient descent."""
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        # compute number of minibatches for training, validation and testing
        num_training_batches = int(size(training_data)/mini_batch_size)
        num_validation_batches = int(size(validation_data)/mini_batch_size)
        num_test_batches = int(size(test_data)/mini_batch_size)

        # define the (regularized) cost function, symbolic gradients, and updates
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self)+\
               0.5*lmbda*l2_norm_squared/num_training_batches
        
        # T.grad == pt.grad
        grads = pt.grad(cost, self.params)
        updates = [(param, param-eta*grad)
                   for param, grad in zip(self.params, grads)]

        # define functions to train a mini-batch, and to compute the
        # accuracy in validation and test mini-batches.
        # T.lscalar == pt.lscalar
        i = pt.lscalar() # mini-batch index
        
        # theano.function == pytensor.function
        train_mb = pytensor.function(
            [i], cost, updates=updates,
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        
        # theano.function == pytensor.function
        validate_mb_accuracy = pytensor.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                validation_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                validation_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        
        # theano.function == pytensor.function
        test_mb_accuracy = pytensor.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        
        # theano.function == pytensor.function
        self.test_mb_predictions = pytensor.function(
            [i], self.layers[-1].y_out,
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        
        # Do the actual training
        best_validation_accuracy = 0.0
        for epoch in range(epochs):
            for minibatch_index in range(num_training_batches):
                iteration = num_training_batches*epoch+minibatch_index
                if iteration % 1000 == 0:
                    print("Training mini-batch number {0}".format(iteration))
                cost_ij = train_mb(minibatch_index)
                if (iteration+1) % num_training_batches == 0:
                    validation_accuracy = np.mean(
                        [validate_mb_accuracy(j) for j in range(num_validation_batches)])
                    print("Epoch {0}: validation accuracy {1:.2%}".format(
                        epoch, validation_accuracy))
                    if validation_accuracy >= best_validation_accuracy:
                        print("This is the best validation accuracy to date.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        if test_data:
                            test_accuracy = np.mean(
                                [test_mb_accuracy(j) for j in range(num_test_batches)])
                            print('The corresponding test accuracy is {0:.2%}'.format(
                                test_accuracy))
        print("Finished training network.")
        print("Best validation accuracy of {0:.2%} obtained at iteration {1}".format(
            best_validation_accuracy, best_iteration))
        print("Corresponding test accuracy of {0:.2%}".format(test_accuracy))

#### Miscellaneous
def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]

def dropout_layer(layer, p_dropout):
    # n = number of trials in the binomial distribution
    # p = probability of success in each trial
    mask = pt.random.binomial(n=1, p=1-p_dropout, size=layer.shape)

    # T.cast == pt.cast; theano.config.floatX == pytensor.config.floatX
    return layer*pt.cast(mask, pytensor.config.floatX)

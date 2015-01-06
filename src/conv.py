"""conv.py
~~~~~~~~~~

Code for many of the experiments involving convolutional networks in
Chapter 6 of the book 'Neural Networks and Deep Learning', by Michael
Nielsen.  The code essentially duplicates (and parallels) what is in
the text, so this is simply a convenience, and has not been commented
in detail.  Consult the original text for more details.

"""

import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
training_data, validation_data, test_data = network3.load_data_shared()
mini_batch_size = 10

def shallow():
    for j in range(3):
        print "A shallow net with 100 hidden neurons"
        net = Network([
            FullyConnectedLayer(n_in=784, n_out=100),
            SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
        net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)

def basic_conv():
    for j in range(3):
        print "Conv + FC architecture"
        net = Network([
            ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                          filter_shape=(20, 1, 5, 5), 
                          poolsize=(2, 2)),
            FullyConnectedLayer(n_in=20*12*12, n_out=100),
            SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
        net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)        

def omit_FC():
    for j in range(3):
        print "Conv only, no FC"
        net = Network([
            ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                          filter_shape=(20, 1, 5, 5), 
                          poolsize=(2, 2)),
            SoftmaxLayer(n_in=20*12*12, n_out=10)], mini_batch_size)
        net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)

def dbl_conv():
    for j in range(3):
        print "Conv + Conv + FC architecture"
        net = Network([
            ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                          filter_shape=(20, 1, 5, 5), 
                          poolsize=(2, 2)),
            ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), 
                          filter_shape=(40, 20, 5, 5), 
                          poolsize=(2, 2)),
            FullyConnectedLayer(n_in=40*4*4, n_out=100),
            SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
        net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)        

def regularized_dbl_conv():
    for lmbda in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        for j in range(3):
            print "Conv + Conv + FC num %s, with regularization %s" % (j, lmbda)
            net = Network([
                ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                              filter_shape=(20, 1, 5, 5), 
                              poolsize=(2, 2)),
                ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), 
                              filter_shape=(40, 20, 5, 5), 
                              poolsize=(2, 2)),
                FullyConnectedLayer(n_in=40*4*4, n_out=100),
                SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
            net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data, lmbda=lmbda)

def dbl_conv_tanh():
    for j in range(3):
        print "Conv + Conv + FC, using tanh, trial %s" % j
        net = Network([
            ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                          filter_shape=(20, 1, 5, 5), 
                          poolsize=(2, 2),
                          activation_fn=tanh),
            ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), 
                          filter_shape=(40, 20, 5, 5), 
                          poolsize=(2, 2),
                          activation_fn=tanh),
            FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=tanh),
            SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
        net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)

def dbl_conv_relu():
    for lmbda in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        for j in range(3):
            print "Conv + Conv + FC num %s, relu, with regularization %s" % (j, lmbda)
            net = Network([
                ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                              filter_shape=(20, 1, 5, 5), 
                              poolsize=(2, 2), 
                              activation_fn=ReLU),
                ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), 
                              filter_shape=(40, 20, 5, 5), 
                              poolsize=(2, 2), 
                              activation_fn=ReLU),
                FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
                SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
            net.SGD(training_data, 60, mini_batch_size, 0.03, validation_data, test_data, lmbda=lmbda)

def expanded_data():
    expanded_training_data, _, _ = network3.load_data_shared(
        "../data/mnist_expanded.pkl.gz")
    for j in range(3):
        print "Training with expanded data, run num %s" % j
        net = Network([
            ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                          filter_shape=(20, 1, 5, 5), 
                          poolsize=(2, 2), 
                          activation_fn=ReLU),
            ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), 
                          filter_shape=(40, 20, 5, 5), 
                          poolsize=(2, 2), 
                          activation_fn=ReLU),
            FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
            SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
        net.SGD(expanded_training_data, 20, mini_batch_size, 0.03, 
                validation_data, test_data, lmbda=1.0)
    

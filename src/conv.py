"""conv.py
~~~~~~~~~~

Code for many of the experiments involving convolutional networks in
Chapter 6 of the book 'Neural Networks and Deep Learning', by Michael
Nielsen.  The code essentially duplicates (and parallels) what is in
the text, so this is simply a convenience, and has not been commented
in detail.  Consult the original text for more details.

"""

from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T

import network3
from network3 import sigmoid, tanh, ReLU, Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

training_data, validation_data, test_data = network3.load_data_shared()
mini_batch_size = 10

def shallow(n=3, epochs=60):
    nets = []
    for j in range(n):
        print "A shallow net with 100 hidden neurons"
        net = Network([
            FullyConnectedLayer(n_in=784, n_out=100),
            SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
        net.SGD(
            training_data, epochs, mini_batch_size, 0.1, 
            validation_data, test_data)
        nets.append(net)
    return nets 

def basic_conv(n=3, epochs=60):
    for j in range(n):
        print "Conv + FC architecture"
        net = Network([
            ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                          filter_shape=(20, 1, 5, 5), 
                          poolsize=(2, 2)),
            FullyConnectedLayer(n_in=20*12*12, n_out=100),
            SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
        net.SGD(
            training_data, epochs, mini_batch_size, 0.1, validation_data, test_data)
    return net 

def omit_FC():
    for j in range(3):
        print "Conv only, no FC"
        net = Network([
            ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                          filter_shape=(20, 1, 5, 5), 
                          poolsize=(2, 2)),
            SoftmaxLayer(n_in=20*12*12, n_out=10)], mini_batch_size)
        net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)
    return net 

def dbl_conv(activation_fn=sigmoid):
    for j in range(3):
        print "Conv + Conv + FC architecture"
        net = Network([
            ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                          filter_shape=(20, 1, 5, 5), 
                          poolsize=(2, 2),
                          activation_fn=activation_fn),
            ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), 
                          filter_shape=(40, 20, 5, 5), 
                          poolsize=(2, 2),
                          activation_fn=activation_fn),
            FullyConnectedLayer(
                n_in=40*4*4, n_out=100, activation_fn=activation_fn),
            SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
        net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)
    return net 

# The following experiment was eventually omitted from the chapter,
# but I've left it in here, since it's an important negative result:
# basic l2 regularization didn't help much.  The reason (I believe) is
# that using convolutional-pooling layers is already a pretty strong
# regularizer.
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

def dbl_conv_relu():
    for lmbda in [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
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

#### Some subsequent functions may make use of the expanded MNIST
#### data.  That can be generated by running expand_mnist.py.

def expanded_data(n=100):
    """n is the number of neurons in the fully-connected layer.  We'll try
    n=100, 300, and 1000.

    """
    expanded_training_data, _, _ = network3.load_data_shared(
        "../data/mnist_expanded.pkl.gz")
    for j in range(3):
        print "Training with expanded data, %s neurons in the FC layer, run num %s" % (n, j)
        net = Network([
            ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                          filter_shape=(20, 1, 5, 5), 
                          poolsize=(2, 2), 
                          activation_fn=ReLU),
            ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), 
                          filter_shape=(40, 20, 5, 5), 
                          poolsize=(2, 2), 
                          activation_fn=ReLU),
            FullyConnectedLayer(n_in=40*4*4, n_out=n, activation_fn=ReLU),
            SoftmaxLayer(n_in=n, n_out=10)], mini_batch_size)
        net.SGD(expanded_training_data, 60, mini_batch_size, 0.03, 
                validation_data, test_data, lmbda=0.1)
    return net 

def expanded_data_double_fc(n=100):
    """n is the number of neurons in both fully-connected layers.  We'll
    try n=100, 300, and 1000.

    """
    expanded_training_data, _, _ = network3.load_data_shared(
        "../data/mnist_expanded.pkl.gz")
    for j in range(3):
        print "Training with expanded data, %s neurons in two FC layers, run num %s" % (n, j)
        net = Network([
            ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                          filter_shape=(20, 1, 5, 5), 
                          poolsize=(2, 2), 
                          activation_fn=ReLU),
            ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), 
                          filter_shape=(40, 20, 5, 5), 
                          poolsize=(2, 2), 
                          activation_fn=ReLU),
            FullyConnectedLayer(n_in=40*4*4, n_out=n, activation_fn=ReLU),
            FullyConnectedLayer(n_in=n, n_out=n, activation_fn=ReLU),
            SoftmaxLayer(n_in=n, n_out=10)], mini_batch_size)
        net.SGD(expanded_training_data, 60, mini_batch_size, 0.03, 
                validation_data, test_data, lmbda=0.1)

def double_fc_dropout(p0, p1, p2, repetitions):
    expanded_training_data, _, _ = network3.load_data_shared(
        "../data/mnist_expanded.pkl.gz")
    nets = []
    for j in range(repetitions):
        print "\n\nTraining using a dropout network with parameters ",p0,p1,p2
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
            FullyConnectedLayer(
                n_in=40*4*4, n_out=1000, activation_fn=ReLU, p_dropout=p0),
            FullyConnectedLayer(
                n_in=1000, n_out=1000, activation_fn=ReLU, p_dropout=p1),
            SoftmaxLayer(n_in=1000, n_out=10, p_dropout=p2)], mini_batch_size)
        net.SGD(expanded_training_data, 40, mini_batch_size, 0.03, 
                validation_data, test_data)
        nets.append(net)
    return nets

def ensemble(nets): 
    """Takes as input a list of nets, and then computes the accuracy on
    the test data when classifications are computed by taking a vote
    amongst the nets.  Returns a tuple containing a list of indices
    for test data which is erroneously classified, and a list of the
    corresponding erroneous predictions.

    Note that this is a quick-and-dirty kluge: it'd be more reusable
    (and faster) to define a Theano function taking the vote.  But
    this works.

    """
    
    test_x, test_y = test_data
    for net in nets:
        i = T.lscalar() # mini-batch index
        net.test_mb_predictions = theano.function(
            [i], net.layers[-1].y_out,
            givens={
                net.x: 
                test_x[i*net.mini_batch_size: (i+1)*net.mini_batch_size]
            })
        net.test_predictions = list(np.concatenate(
            [net.test_mb_predictions(i) for i in xrange(1000)]))
    all_test_predictions = zip(*[net.test_predictions for net in nets])
    def plurality(p): return Counter(p).most_common(1)[0][0]
    plurality_test_predictions = [plurality(p) 
                                  for p in all_test_predictions]
    test_y_eval = test_y.eval()
    error_locations = [j for j in xrange(10000) 
                       if plurality_test_predictions[j] != test_y_eval[j]]
    erroneous_predictions = [plurality(all_test_predictions[j])
                             for j in error_locations]
    print "Accuracy is {:.2%}".format((1-len(error_locations)/10000.0))
    return error_locations, erroneous_predictions

def plot_errors(error_locations, erroneous_predictions=None):
    test_x, test_y = test_data[0].eval(), test_data[1].eval()
    fig = plt.figure()
    error_images = [np.array(test_x[i]).reshape(28, -1) for i in error_locations]
    n = min(40, len(error_locations))
    for j in range(n):
        ax = plt.subplot2grid((5, 8), (j/8, j % 8))
        ax.matshow(error_images[j], cmap = matplotlib.cm.binary)
        ax.text(24, 5, test_y[error_locations[j]])
        if erroneous_predictions:
            ax.text(24, 24, erroneous_predictions[j])
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.tight_layout()
    return plt
    
def plot_filters(net, layer, x, y):

    """Plot the filters for net after the (convolutional) layer number
    layer.  They are plotted in x by y format.  So, for example, if we
    have 20 filters after layer 0, then we can call show_filters(net, 0, 5, 4) to
    get a 5 by 4 plot of all filters."""
    filters = net.layers[layer].w.eval()
    fig = plt.figure()
    for j in range(len(filters)):
        ax = fig.add_subplot(y, x, j)
        ax.matshow(filters[j][0], cmap = matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.tight_layout()
    return plt


#### Helper method to run all experiments in the book

def run_experiments():

    """Run the experiments described in the book.  Note that the later
    experiments require access to the expanded training data, which
    can be generated by running expand_mnist.py.

    """
    shallow()
    basic_conv()
    omit_FC()
    dbl_conv(activation_fn=sigmoid)
    # omitted, but still interesting: regularized_dbl_conv()
    dbl_conv_relu()
    expanded_data(n=100)
    expanded_data(n=300)
    expanded_data(n=1000)
    expanded_data_double_fc(n=100)    
    expanded_data_double_fc(n=300)
    expanded_data_double_fc(n=1000)
    nets = double_fc_dropout(0.5, 0.5, 0.5, 5)
    # plot the erroneous digits in the ensemble of nets just trained
    error_locations, erroneous_predictions = ensemble(nets)
    plt = plot_errors(error_locations, erroneous_predictions)
    plt.savefig("ensemble_errors.png")
    # plot the filters learned by the first of the nets just trained
    plt = plot_filters(nets[0], 0, 5, 4)
    plt.savefig("net_full_layer_0.png")
    plt = plot_filters(nets[0], 1, 8, 5)
    plt.savefig("net_full_layer_1.png")


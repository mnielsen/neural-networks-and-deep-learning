"""
common_knowledge
~~~~~~~~~~~~~~~~

Try to determine whether or not it's possible to relate the
descriptions given by two different autoencoders.

"""

#### Libraries
# My libraries
from backprop2 import Network, sigmoid_vec
import mnist_loader

# Third-party libraries
import numpy as np


#### Parameters
# Size of the training sets.  May range from 1000 to 12,500.  Lower
# will be faster, higher will give more accuracy.
SIZE = 1000 


# Generate training sets
training_data, _, _ = mnist_loader.load_data_nn()
td_1 = [(x, x) for x, _ in training_data[0:SIZE]]
td_2 = [(x, x) for x, _ in training_data[12500:12500+SIZE]]
td_3 = [x for x, _ in training_data[25000:25000+SIZE]]
test = [x for x, _ in training_data[37500:37500+SIZE]]

print "\nFinding first autoencoder"
ae_1 = Network([784, 30, 784])
ae_1.SGD(td_1, 4, 10, 0.01, 0.05)

print "\nFinding second autoencoder"
ae_2 = Network([784, 30, 784])
ae_2.SGD(td_1, 4, 10, 0.01, 0.05)

print "\nGenerating encoded training data"
encoded_td_1 = [sigmoid_vec(np.dot(ae_1.weights[0], x)+ae_1.biases[0])
                for x in td_3]
encoded_td_2 = [sigmoid_vec(np.dot(ae_2.weights[0], x)+ae_2.biases[0])
                for x in td_3]
encoded_training_data = zip(encoded_td_1, encoded_td_2)

print "\Finding mapping between theories"
net = Network([30, 60, 30])
net.SGD(encoded_training_data, 6, 10, 0.01, 0.05)


print "\nComparing theories"
encoded_test_1 = [sigmoid_vec(np.dot(ae_1.weights[0], x)+ae_1.biases[0])
                  for x in test]
encoded_test_2 = [sigmoid_vec(np.dot(ae_2.weights[0], x)+ae_2.biases[0])
                  for x in test]
test_data = zip(encoded_test_1, encoded_test_2)
print "Mean desired output activation: %s" % (
    sum(y.mean() for _, y in test_data) / SIZE,)
error = sum([np.sum((net.feedforward(x)-y)**2) for (x, y) in test_data])
print "Mean square error per training image: %s" % (error / SIZE,)

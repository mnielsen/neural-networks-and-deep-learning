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
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


#### Parameters
# Size of the training sets.  May range from 1000 to 12,500.  Lower
# will be faster, higher will give more accuracy.
SIZE = 5000 
# Number of hidden units in the autoencoder
HIDDEN = 30

print "\nGenerating training data"
training_data, _, _ = mnist_loader.load_data_nn()
td_1 = [(x, x) for x, _ in training_data[0:SIZE]]
td_2 = [(x, x) for x, _ in training_data[12500:12500+SIZE]]
td_3 = [x for x, _ in training_data[25000:25000+SIZE]]
test = [x for x, _ in training_data[37500:37500+SIZE]]

print "\nFinding first autoencoder"
ae_1 = Network([784, HIDDEN, 784])
ae_1.SGD(td_1, 4, 10, 0.01, 0.05)

print "\nFinding second autoencoder"
ae_2 = Network([784, HIDDEN, 784])
ae_2.SGD(td_1, 4, 10, 0.01, 0.05)

print "\nGenerating encoded training data"
encoded_td_1 = [sigmoid_vec(np.dot(ae_1.weights[0], x)+ae_1.biases[0])
                for x in td_3]
encoded_td_2 = [sigmoid_vec(np.dot(ae_2.weights[0], x)+ae_2.biases[0])
                for x in td_3]
encoded_training_data = zip(encoded_td_1, encoded_td_2)

print "\nFinding mapping between theories"
net = Network([HIDDEN, HIDDEN])
net.SGD(encoded_training_data, 6, 10, 0.01, 0.05)

print """\nBaseline for comparison: decompress with the first autoencoder"""
print """and compress with the second autoencoder"""
encoded_test_1 = [sigmoid_vec(np.dot(ae_1.weights[0], x)+ae_1.biases[0])
                  for x in test]
encoded_test_2 = [sigmoid_vec(np.dot(ae_2.weights[0], x)+ae_2.biases[0])
                  for x in test]
test_data = zip(encoded_test_1, encoded_test_2)
net_baseline = Network([HIDDEN, 784, HIDDEN])
net_baseline.biases[0] = ae_1.biases[1]
net_baseline.weights[0] = ae_1.weights[1]
net_baseline.biases[1] = ae_2.biases[0]
net_baseline.weights[1] = ae_2.weights[0]
error_baseline = sum(np.linalg.norm(net_baseline.feedforward(x)-y, 1) 
                     for (x, y) in test_data)
print "Baseline average l1 error per training image: %s" % (error_baseline / SIZE,)

print "\nComparing theories with a simple interconversion"
print "Mean desired output activation: %s" % (
    sum(y.mean() for _, y in test_data) / SIZE,)
error = sum(np.linalg.norm(net.feedforward(x)-y, 1) for (x, y) in test_data)
print "Average l1 error per training image: %s" % (error / SIZE,)

print "\nComputing fiducial image inputs"
fiducial_images_1 = [
    ae_1.weights[0][j,:].reshape(28,28)/np.linalg.norm(net.weights[0][j,:])
    for j in range(HIDDEN)]
fiducial_images_2 = [
    ae_2.weights[0][j,:].reshape(28,28)/np.linalg.norm(net.weights[0][j,:])
    for j in range(HIDDEN)]
image = np.concatenate([np.concatenate(fiducial_images_1, axis=1), 
                        np.concatenate(fiducial_images_2, axis=1)])
fig = plt.figure()
ax = fig.add_subplot(111)
ax.matshow(image, cmap = matplotlib.cm.binary)
plt.xticks(np.array([]))
plt.yticks(np.array([]))
plt.show()

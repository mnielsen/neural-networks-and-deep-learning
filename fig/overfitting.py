"""
overfitting
~~~~~~~~~~~

Plot graphs to illustrate the problem of overfitting.
"""

# Standard library
import imp

# My library
mnist_loader = imp.load_source('mnist_loader', '../code/mnist_loader.py')
network2 = imp.load_source('network2', '../code/network2.py')

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np


# Generate the results
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost())
evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = \
    net.SGD(training_data[:1000], 150, 10, 0.05,
            evaluation_data=test_data, 
            monitor_evaluation_cost=True, 
            monitor_evaluation_accuracy=True, 
            monitor_training_cost=True, 
            monitor_training_accuracy=True)

# Do the plots

#Plot the training cost data
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(40, 150, 1), training_cost[40:150])
ax.set_xlim([40, 150])
ax.grid(True)
ax.set_xlabel('Epoch')
ax.set_title('Cost on the training data')
plt.show()

# Plot the test accuracy data
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(40, 150, 1), 
        [accuracy/100.0 for accuracy in evaluation_accuracy[40:150]])
ax.set_xlim([40, 150])
ax.grid(True)
ax.set_xlabel('Epoch')
ax.set_title('Accuracy (%) on the test data')
plt.show()

#Plot the test cost data
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(40, 150, 1), evaluation_cost[40:150])
ax.set_xlim([40, 150])
ax.grid(True)
ax.set_xlabel('Epoch')
ax.set_title('Cost on the test data')
plt.show()

# Plot the training accuracy data
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(0, 150, 1), [accuracy/100.0 for accuracy in training_accuracy[40:150]])
ax.set_xlim([0, 150])
ax.grid(True)
ax.set_xlabel('Epoch')
ax.set_title('Accuracy (%) on the training data')
plt.show()

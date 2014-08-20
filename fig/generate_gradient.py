"""generate_gradient.py
~~~~~~~~~~~~~~~~~~~~~~~

Use network2 to figure out the average starting values of the gradient
error terms \delta^l_j = \partial C / \partial z^l_j = \partial C /
\partial b^l_j.

"""

import numpy as np

import sys
sys.path.append("../code/")
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

import network2
net = network2.Network([784, 30, 30, 10])
nabla_b_results = [net.backprop(x, y)[0] for x, y in training_data[:1000]]
def sum(a, b): return [x+y for (x, y) in zip(a, b)]
gradient = reduce(sum, nabla_b_results)
average_gradient = [(np.reshape(g, len(g))/1000).tolist() for g in gradient]
# Discard all but the first 6 terms in each layer, discard the output layer
abbreviated_gradient = [ag[:6] for ag in average_gradient[:-1]] 

import json
f = open("initial_gradient.json", "w")
json.dump(abbreviated_gradient, f)
f.close()

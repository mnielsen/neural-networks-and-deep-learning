import light_format_feature_loader
import network

training_data, validation_data, test_data = light_format_feature_loader.load_data_wrapper()

""" 
Build a three-layer neural network given the number of neurons for each layer
 Parameters:
 [number of neurons for input layer], [number of neurons for hidden layer], [number of neurons for output layer]
"""
net = network.Network([784, 30, 10])

"""
Perform stochastic gradient descent algorithm given necessary hyper-parameters
 Parameters:
 [training_data], [number of epoches], [mini-batch size], [training rate], [test_data]
 net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
"""
net.SGD(training_data, 30, 10, 10.0, test_data=test_data)



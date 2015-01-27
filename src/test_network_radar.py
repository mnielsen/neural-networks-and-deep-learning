import light_format_feature_loader
import network

rainfall_min_value = 0
rainfall_max_value = 100
n_rainfall_indexes = 100 # for use in vectorized form of rainfall label
training_data, validation_data, test_data, dimension = light_format_feature_loader.load_data_wrapper(rainfall_min_value, rainfall_max_value, n_rainfall_indexes)

""" 
Build a three-layer neural network given the number of neurons for each layer
 Parameters:
 [number of neurons for input layer], [number of neurons for hidden layer], [number of neurons for output layer]
"""
net = network.Network([dimension, 30, n_rainfall_indexes])

"""
Perform stochastic gradient descent algorithm given necessary hyper-parameters
 Parameters:
 [training_data], [number of epoches], [mini-batch size], [training rate], [test_data]
 net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
"""
net.SGD(training_data, 1000, 10, 1.0, test_data=test_data)



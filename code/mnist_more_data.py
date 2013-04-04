"""
mnist_more_data
~~~~~~~~~~~~~~~

This program produces a plot showing how the size of the training data
set affects the classification accuracy of an SVM and a neural network
classifier.  The training and test data is drawn from the MNIST data
set.
"""

#### Libraries
# My libraries
import mnist_nn
import mnist_loader 

# Third-party libraries
import matplotlib
import matplotlib.pyplot as plt
from sklearn import svm

svm_training_data, _, svm_test_data = mnist_loader.load_data()
nn_training_data, nn_test_inputs, nn_actual_test_results = mnist_nn.load_data()
sizes = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000] 
svm_results, nn_results = [], []
for size in sizes:
    print "\nData set size: %s" % size
    # SVM results
    clf = svm.SVC()
    clf.fit(svm_training_data[0][:size], svm_training_data[1][:size])
    predictions = [int(a) for a in clf.predict(svm_test_data[0])]
    svm_results.append(
        sum(int(a == y) for a, y in zip(predictions, svm_test_data[1])))
    print "SVM result: %s /  10000" % svm_results[-1]
    # Neural network results
    net = mnist_nn.Network([784, 20, 10])
    epochs = 1500000/size
    net.SGD(nn_training_data[:size], epochs, 10, 0.01, 0.001)
    nn_results.append(net.evaluate(nn_test_inputs, nn_actual_test_results))
    print "Neural net result: %s / 10000" % nn_results[-1]

plt.semilogx(sizes, svm_results, 'bo-', sizes, nn_results, 'ro-')
plt.axis([80, 60000, 0, 10000])
plt.xlabel("Size")
plt.ylabel("Accuracy")
plt.legend(("SVM", "Neural net"), "lower left")
plt.show()

"""
mnist_svm
~~~~~~~~~

A classifier program for recognizing handwritten digits from the MNIST
data set, using an SVM classifier.

The program first trains the classifier, and then applies the
classifier to the MNIST test data to see how many digits are correctly
classified.  Finally, the program uses matplotlib to create an image
of the first 10 digits which are incorrectly classified."""

#### Libraries
# My libraries
import mnist_loader # to load the MNIST data.  For details on the
                    # format the data is loaded in, see the module's
                    # code

# Third-party libraries
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

def svm_baseline():
    training_data, validation_data, test_data = mnist_loader.load_data()
    # train
    clf = svm.SVC()
    clf.fit(training_data[0], training_data[1])
    # test
    predictions = [int(a) for a in clf.predict(test_data[0])]
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))
    print "Baseline classifier using an SVM."
    print "%s of %s values correct." % (num_correct, len(test_data[1]))
    # finally, plot the first ten images where the classifier fails
    failure_indices = [j for (j, z) in enumerate(zip(predictions, test_data[1]))
                       if z[0] != z[1]]
    failed_images = [np.reshape(test_data[0][failure_indices[j]], (-1, 28))
                     for j in xrange(10)]
    fig = plt.figure()
    for j in xrange(1, 11):
        ax = fig.add_subplot(1, 10, j)
        ax.matshow(failed_images[j-1], cmap = matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.show()

if __name__ == "__main__":
    svm_baseline()
    

"""
mnist
~~~~~

Draws images based on the MNIST data.

Note that the code requires the file ``../../data/mnist.pkl``.  This is not
included in the repository.  It may be downloaded from:

http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
"""

#### Libraries
# Standard library
import cPickle

# Third-party libraries
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def main():
    training_set, validation_set, test_set = load_data()
    images = get_images(training_set)[:100]
    plot_10_by_10_images(images)

#### Plotting
def plot_images_together(images):
    """ Plot a single image containing all six MNIST images, one after
    the other.  Note that we crop the sides of the images so that they
    appear reasonably close together."""
    fig = plt.figure()
    images = [image[:, 3:25] for image in images]
    image = np.concatenate(images, axis=1)
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(image, cmap = matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.show()

def plot_10_by_10_images(images):
    """ Plot 100 MNIST images in a 10 by 10 table. Note that we crop
    the images so that they appear reasonably close together.  The
    image is post-processed to give the appearance of being continued."""
    fig = plt.figure()
    images = [image[3:25, 3:25] for image in images]
    #image = np.concatenate(images, axis=1)
    for x in range(10):
        for y in range(10):
            ax = fig.add_subplot(10, 10, 10*y+x)
            ax.matshow(images[10*y+x], cmap = matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
    plt.show()

def plot_images_separately(images):
    "Plot the six MNIST images separately."
    fig = plt.figure()
    for j in xrange(1, 7):
        ax = fig.add_subplot(1, 6, j)
        ax.matshow(images[j-1], cmap = matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.show()

def plot_mnist_digit(image):
    """ Plot a single MNIST image."""
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(image, cmap = matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.show()

def plot_2_and_1(images):
    "Plot a 2 and a 1 image from the MNIST set."
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.matshow(images[5], cmap = matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    ax = fig.add_subplot(1, 2, 2)
    ax.matshow(images[3], cmap = matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.show()

def plot_top_left(image):
    "Plot the top left of ``image``."
    image[14:,:] = np.zeros((14,28))
    image[:,14:] = np.zeros((28,14))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(image, cmap = matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.show()

def plot_features(image):
    "Plot the top right, bottom left, and bottom right of ``image``."
    image_1, image_2, image_3 = np.copy(image), np.copy(image), np.copy(image)
    image_1[:,:14] = np.zeros((28,14))
    image_1[14:,:] = np.zeros((14,28))
    image_2[:,14:] = np.zeros((28,14))
    image_2[:14,:] = np.zeros((14,28))
    image_3[:14,:] = np.zeros((14,28))
    image_3[:,:14] = np.zeros((28,14))
    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    ax.matshow(image_1, cmap = matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    ax = fig.add_subplot(1, 3, 2)
    ax.matshow(image_2, cmap = matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    ax = fig.add_subplot(1, 3, 3)
    ax.matshow(image_3, cmap = matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.show()

#### Miscellanea
def load_data():
    """ Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data."""
    f = open('../../data/mnist.pkl', 'rb')
    training_set, validation_set, test_set = cPickle.load(f)
    f.close()
    return (training_set, validation_set, test_set)

def get_images(training_set):
    """ Return a list containing the first 100 images from the MNIST
    data set. Each image is represented as a 2-d numpy array."""
    flattened_images = training_set[0][:100]
    return [np.reshape(f, (-1, 28)) for f in flattened_images]

#### Main
if __name__ == "__main__":
    main()

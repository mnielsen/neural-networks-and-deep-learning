"""
mnist
~~~~~

Draws images based on the MNIST data."""

#### Libraries
# Standard library
import cPickle

# Third-party libraries
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def main():
    training_set, validation_set, test_set = load_data()
    images = get_images()
    plot_images_together(images)
    plot_images_separately(images)

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

def plot_images_separately(images):
    "Plot the six MNIST images separately."
    fig = plt.figure()
    for j in xrange(1, 7):
        ax = fig.add_subplot(1, 6, j)
        ax.matshow(images[j-1], cmap = matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.show()

#### Miscellanea
def load_data():
    """ Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data."""
    f = open('mnist.pkl', 'rb')
    training_set, validation_set, test_set = cPickle.load(f)
    f.close()
    return (training_set, validation_set, test_set)

def get_images():
    """ Return a list containing the first six images from the MNIST
    data set. Each image is represented as a 2-d numpy array."""
    flattened_images = training_set[0][:6]
    return [np.reshape(f, (-1, 28)) for f in flattened_images]

#### Main
if __name__ == "__main__":
    main()

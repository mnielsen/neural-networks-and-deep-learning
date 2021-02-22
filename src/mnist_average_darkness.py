"""
mnist_average_darkness
~~~~~~~~~~~~~~~~~~~~~~

A naive classifier for recognizing handwritten digits from the MNIST
data set.  The program classifies digits based on how dark they are
--- the idea is that digits like "1" tend to be less dark than digits
like "8", simply because the latter has a more complex shape.  When
shown an image the classifier returns whichever digit in the training
data had the closest average darkness.

The program works in two steps: first it trains the classifier, and
then it applies the classifier to the MNIST test data to see how many
digits are correctly classified.

Needless to say, this isn't a very good way of recognizing handwritten
digits!  Still, it's useful to show what sort of performance we get
from naive ideas."""

#### Libraries
# Standard library
from collections import defaultdict

# My libraries
import mnist_loader

def main():
    training_data, validation_data, test_data = mnist_loader.load_data()
    # training phase: compute the average darknesses for each digit,
    # based on the training data
    avgs = avg_darknesses(training_data)
    # testing phase: see how many of the test images are classified
    # correctly
    num_correct = sum(int(guess_digit(image, avgs) == digit)
                      for image, digit in zip(test_data[0], test_data[1]))
    print("Baseline classifier using average darkness of image.")
    print("{0} of {1} values correct.".format(num_correct, len(test_data[1])))

def avg_darknesses(training_data):
    """ Return a defaultdict whose keys are the digits 0 through 9.
    For each digit we compute a value which is the average darkness of
    training images containing that digit.  The darkness for any
    particular image is just the sum of the darknesses for each pixel."""
    digit_counts = defaultdict(int)
    darknesses = defaultdict(float)
    for image, digit in zip(training_data[0], training_data[1]):
        digit_counts[digit] += 1
        darknesses[digit] += sum(image)
    avgs = defaultdict(float)
    for digit, n in digit_counts.items():
        avgs[digit] = darknesses[digit] / n
    return avgs

def guess_digit(image, avgs):
    """Return the digit whose average darkness in the training data is
    closest to the darkness of ``image``.  Note that ``avgs`` is
    assumed to be a defaultdict whose keys are 0...9, and whose values
    are the corresponding average darknesses across the training data."""
    darkness = sum(image)
    distances = {k: abs(v-darkness) for k, v in avgs.items()}
    return min(distances, key=distances.get)

if __name__ == "__main__":
    main()

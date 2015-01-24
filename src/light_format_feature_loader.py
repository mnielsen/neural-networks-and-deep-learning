"""
light_format_feature_loader
~~~~~~~~~~~~

A library to load light format feature data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library

# Third-party libraries
import numpy as np


def parse_line(line):
    tokens = str(line).strip().split()
    #print tokens
    label = tokens[0]
    feature_vector = []
    dimension = len(tokens) - 1
    #print "dimension: " + str(dimension)
    #print "label: " + str(label)
    for i in range(1, dimension+1):
        # print tokens[i]
        index_value_pair = str(tokens[i]).split(":")
        feature_vector.append(float(index_value_pair[1]))
    # print "feature_vector: " + str(feature_vector)
    return (label, feature_vector, dimension)

def load_data():
    """Return light format feature data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training feature vectors.
    This is a numpy ndarray with feature vectors/entries read from the input file. 
    Each feature vector/entry is, in turn, also a numpy ndarray.

    The second entry in the ``training_data`` tuple is a numpy ndarray.
    Each of the entries are labels for the corresponding
    feature vectors contained in the first entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    
    training_data_dir = "/home/awips/sample_data/svmdata.oneHourPrecipAsLabel/svm_case1_merged"
    training_data_file = "svm_traindata.txt.1a.Z_0.5_0.5"
    print training_data_dir + "/" + training_data_file
    f = open(training_data_dir + "/" + training_data_file, "r")
    lines = f.readlines()
    feature_collection = []
    for line in lines:
        (_label, _feature_vector, _dimension) = parse_line(line)
        feature_collection.append(_feature_vector)
        
    for feature_vector in feature_collection:
        print "feature_vector: " + str(feature_vector)
    
    # Not implemented yet
    #f = gzip.open('../data/mnist.pkl.gz', 'rb')
    #training_data, validation_data, test_data = cPickle.load(f)
    #f.close()
    #return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list of 2-tuples ``(x, y)``.  
    ``x`` is a xxx-dimensional numpy.ndarray containing the input feature vector. 
    ``y`` is a xx-dimensional numpy.ndarray representing the unit vector corresponding
    to the label for ``x``.

    ``validation_data`` and ``test_data`` are lists containing xxx
    2-tuples ``(x, y)``.  In each case, ``x`` is a xx-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data()
    
    # While tr_d[0] is a ndarray containing the feature vectors/entries, each entry is also a ndarray.
    # it looks like a list of vectors when printing out, 
    # each vector is xx-dimensional, also looks like in the form of a list
    print "#####################################################################"
    print "# Part I: Training data before being transformed for neural network #"
    print "#####################################################################"
    print "tr_d.__len__(): " + str(tr_d.__len__())
    print "=== tr_d[0] is info about training feature vectors ==="
    print "tr_d[0]: "
    print tr_d[0]
    count = 0
    print "tr_d[0][0].size (size/dimension of the first feature vector): " + str(tr_d[0][0].size)
    print "tr_d[0][1].size (size/dimension of the second feature vector): " + str(tr_d[0][1].size)
    print "....."
    for i in tr_d[0]:
        count = count + 1
        #print i
        #print i.size
        #break
    print "count (number of feature vectors): " + str(count)
    print "count * tr_d[0][0].size: " + str(count) + " * " + str(tr_d[0][0].size) + " = " + str(count * tr_d[0][0].size)
    print "ie. tr_d[0].size: " + str(tr_d[0].size)
    print ""
    
    # tr_d[1] is a ndarray containing the label entries corresponding to the feature vectors, 
    # each entry is just the scalar  label/result value
    print "=== tr_d[1] is info about corresponding labels/results ==="
    print "tr_d[1]: "
    print tr_d[1]
    print "tr_d[1][0] (rainfall value for the 1st feature vector): " + str(tr_d[1][0]);
    print "tr_d[1][1] (rainfall value for the 2nd feature vector): " + str(tr_d[1][1]);
    print "....."
    count_labels = 0
    for i in tr_d[1]:
        count_labels = count_labels + 1
        #print i
        #print i.size
        #break
    print "count_labels (number of labels): " + str(count_labels)
    print "count * tr_d[1][0].size: " + str(count) + " * " + str(tr_d[1][0].size) + " = "+ str(count*tr_d[1][0].size);
    print "ie. tr_d[1].size: " + str(tr_d[1].size)

    print ""
    print ""
    print "###############################################################################"
    print "# Part II: Training data after being transformed for neural network as inputs #"
    print "###############################################################################"
    print "=== Transformed training_inputs ==="
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    print "training_inputs.__len__():" + str(training_inputs.__len__())
    # print "training_inputs: "
    for input in training_inputs:
        print "single_training_input.size: " + str(input.size)
        # print input
        break # Print only the first input feature vector/image
    
    training_results = [vectorized_result(y) for y in tr_d[1]]
    for result in training_results:
        print "single_vectorized_result.size (label/digit): " + str(result.size)
        print result
        break # Print only the first result(Or label, digit)
    
    print "=== training_data (zip of transformed training_inputs and corresponding labels) ==="
    training_data = zip(training_inputs, training_results)
    for data in training_data:
        print "single_training_data: " + str(data)
        # print data
        break # Print only the first input feature vector followed by its corresponding labels/digits

    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

if __name__ == '__main__':
    import light_format_feature_loader
    # training_data, validation_data, test_data = light_format_feature_loader.load_data_wrapper()
    # tr_d, va_d, te_d = light_format_feature_loader.load_data()
    light_format_feature_loader.load_data()


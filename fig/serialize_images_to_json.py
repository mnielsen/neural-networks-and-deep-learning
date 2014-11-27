"""
serialize_images_to_json
~~~~~~~~~~~~~~~~~~~~~~~~

Utility to serialize parts of the training and validation data to JSON, 
for use with Javascript.  """

#### Libraries
# Standard library
import json 
import sys

# My library
sys.path.append('../src/')
import mnist_loader

# Third-party libraries
import numpy as np


# Number of training and validation data images to serialize
NTD = 1000
NVD = 100

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

def make_data_integer(td):
    # This will be slow, due to the loop.  It'd be better if numpy did
    # this directly.  But numpy.rint followed by tolist() doesn't
    # convert to a standard Python int.
    return [int(x) for x in (td*256).reshape(784).tolist()]

data = {"training": [
    {"x": [x[0] for x in training_data[j][0].tolist()],
     "y": [y[0] for y in training_data[j][1].tolist()]}
    for j in xrange(NTD)],
        "validation": [
    {"x": [x[0] for x in validation_data[j][0].tolist()],
     "y": validation_data[j][1]}
            for j in xrange(NVD)]}

f = open("data_1000.json", "w")
json.dump(data, f)
f.close()



"""
mnist_pca
~~~~~~~~~

Use PCA to reconstruct some of the MNIST test digits.
"""

# My libraries
import mnist_loader

# Third-party libraries
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import RandomizedPCA


# Training
training_data, test_inputs, actual_test_results = mnist_loader.load_data_nn()
pca = RandomizedPCA(n_components=30)
nn_images = [x for (x, y) in training_data]
pca_images = np.concatenate(nn_images, axis=1).transpose()
pca_r = pca.fit(pca_images)

# Try PCA on first ten test images
test_images = np.array(test_inputs[:10]).reshape((10,784))
test_outputs = pca_r.inverse_transform(pca_r.transform(test_images))

# Plot the first ten test images and the corresponding outputs
fig = plt.figure()
ax = fig.add_subplot(111)
images_in = [test_inputs[j].reshape(-1, 28) for j in range(10)]
images_out = [test_outputs[j].reshape(-1, 28) for j in range(10)]
image_in = np.concatenate(images_in, axis=1)
image_out = np.concatenate(images_out, axis=1)
image = np.concatenate([image_in, image_out])
ax.matshow(image, cmap = matplotlib.cm.binary)
plt.xticks(np.array([]))
plt.yticks(np.array([]))
plt.show()

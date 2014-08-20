"""
pca_limitations
~~~~~~~~~~~~~~~

Plot graphs to illustrate the limitations of PCA.
"""

# Third-party libraries
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# Plot just the data
fig = plt.figure()
ax = fig.gca(projection='3d')
z = np.linspace(-2, 2, 20)
theta = np.linspace(-4 * np.pi, 4 * np.pi, 20)
x = np.sin(theta)+0.03*np.random.randn(20)
y = np.cos(theta)+0.03*np.random.randn(20)
ax.plot(x, y, z, 'ro')
plt.show()

# Plot the data and the helix together
fig = plt.figure()
ax = fig.gca(projection='3d')
z_helix = np.linspace(-2, 2, 100)
theta_helix = np.linspace(-4 * np.pi, 4 * np.pi, 100)
x_helix = np.sin(theta_helix)
y_helix = np.cos(theta_helix)
ax.plot(x, y, z, 'ro')
ax.plot(x_helix, y_helix, z_helix, '')
plt.show()

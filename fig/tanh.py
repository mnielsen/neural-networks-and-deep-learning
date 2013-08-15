"""
tanh
~~~~

Plots a graph of the tanh function."""

import numpy as np
import matplotlib.pyplot as plt

z = np.arange(-5, 5, .1)
t = np.tanh(z)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(z, t)
ax.set_ylim([-1.0, 1.0])
ax.set_xlim([-5,5])
ax.grid(True)
ax.set_xlabel('z')
ax.set_title('tanh function')

plt.show()

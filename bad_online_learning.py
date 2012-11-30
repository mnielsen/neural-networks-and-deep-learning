"""
bad_online_learning
~~~~~~~~~~~~~~~~~~~

A contour plot showing an instance where online learning may fail."""

#### Libraries
# Third party libraries
import matplotlib.pyplot as plt
import numpy

X = numpy.arange(-1, 1, 0.05)
Y = numpy.arange(-1, 1, 0.05)
X, Y = numpy.meshgrid(X, Y)
Z = -X**2 - Y**2 + numpy.exp(-4*X**2+0.3*Y+X)

plt.figure()
CS = plt.contour(X, Y, Z)
plt.show()

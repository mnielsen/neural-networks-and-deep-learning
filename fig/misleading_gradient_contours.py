"""
misleading_gradient_contours
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plots the contours of the function from misleading_gradient.py"""

#### Libraries
# Third party libraries
import matplotlib.pyplot as plt
import numpy

X = numpy.arange(-1, 1, 0.02)
Y = numpy.arange(-1, 1, 0.02)
X, Y = numpy.meshgrid(X, Y)
Z = X**2 + 10*Y**2

plt.figure()
CS = plt.contour(X, Y, Z, levels=[0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
plt.xlabel("$w_1$", fontsize=16)
plt.ylabel("$w_2$", fontsize=16)
plt.show()

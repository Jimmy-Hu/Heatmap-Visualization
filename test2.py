import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

X = np.arange(-5, 5, 0.1)
Y = np.arange(-5, 5, 0.25)
print('X: {}, Y: {}'.format(X.shape, Y.shape))
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

print('X: {}, Y: {}, Z: {}'.format(X.shape, Y.shape, Z.shape))
print(X)
print(Y)
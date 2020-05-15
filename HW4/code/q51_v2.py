import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f(x,y):
    return 0.2*x*y - np.cos(y)

def grad_f(x,y):
    grad_x = 0.2*y
    grad_y = 0.2*x + np.sin(y)

    return grad_x, grad_y

num_points = 1000
X = np.arange(-1,1,2/num_points)
Y = np.arange(-2*np.pi, 2*np.pi, 4.*np.pi/num_points)

Y_maxs = []
Z_Y_maxs = []

for x in X:
    Zs = f(x*np.ones_like(Y), Y)
    z_max_id = np.argmax(Zs)
    Y_maxs.append(Y[z_max_id]*1.0)
    Z_Y_maxs.append(Zs[z_max_id]*1.0)

X_star = X[np.argmin(Z_Y_maxs)]
Y_star = Y_maxs[np.argmin(Z_Y_maxs)]

Z_star = f(X_star, Y_star)




print("(x*,y*,Z*) ",(X_star, Y_star, Z_star))



grad_x, grad_y = grad_f(X_star, Y_star)
print("(grad_x*,grad_y*) ",(grad_x, grad_y))

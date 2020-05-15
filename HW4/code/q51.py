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

num_points = 10000
X = np.linspace(-1,1,num_points)
Y = np.linspace(-2*np.pi, 2*np.pi, num_points)

XX, YY = np.meshgrid(X, Y)

Z = f(XX, YY)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# ax.plot3D(X,Y,F)
# ax.contour3D(X, Y, Z, 100)


Y_maxs = np.argmax(Z, axis=0)
X_star_id = np.argmin(Y_maxs)
Y_star_id = Y_maxs[X_star_id]
X_star = X[X_star_id]
Y_star = Y[Y_star_id]
Z_star = Z[Y_star_id, X_star_id]

# ax.scatter(X_star, Y_star, Z_star)
# plt.show()



print("(x*,y*,Z*) ",(X_star, Y_star, Z_star))

print(f(0, np.pi))


grad_x, grad_y = grad_f(X_star, Y_star)
print("(grad_x*,grad_y*) ",(grad_x, grad_y))

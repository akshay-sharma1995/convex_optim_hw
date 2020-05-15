import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
import pdb

def objective_func(x):
    return x + np.sin(6*x)

class GPR():
    def __init__(self, var_K):
        self.var_K = var_K
        self.M = None
        

    def K(self, x,y):
        norm_xy = np.linalg.norm(x-y, ord=2, axis=-1)
        return np.exp(-(norm_xy**2) / (2*self.var_K))
    
    def fit(self,X, Y):
        M = np.zeros(shape=(X.shape[0], X.shape[0]))

        for i,x in enumerate(X[:,0]):
            M[i,:] = self.K(x*np.ones_like(X), X)

        self.M = M
        self.X = X
        self.Y = Y
        
    def predict(self, X, return_std = False):
        
        means = []
        stds = []
        vals = []
        for x in X:
            v = self.K(x*np.ones_like(self.X),self.X).reshape(-1,1)
            x_arr = np.array(x).reshape(-1,1)
            K_x = self.K(x_arr, x_arr)
            
            temp = np.matmul(v.T, np.linalg.inv(self.M))
            
            mu = np.matmul(temp, self.Y)
            var = K_x - np.matmul(temp, v)
            if(var<0):
                var = np.array([0]).reshape(var.shape)
            val = np.random.normal(mu, var**0.5, 1)
            vals.append(val*1.0)
            means.append(mu[0]*1.0)
            stds.append(var[0]**0.5)
        
        means = np.array(means)
        vals = np.array(vals).reshape(means.shape)
        stds = np.array(stds)
        if return_std:
            return vals, stds
        else:
            return vals

        
def surrogate_func(model, X):
    return model.predict(X, return_std=True)

def acquisition(X, Xsamples, model):
    yhat, _ = surrogate_func(model, X)
    best = max(yhat)
    mu, std = surrogate_func(model, Xsamples)
    mu = mu
    ei = (mu - best)*norm.cdf((mu-best)/std) + (std/(2*np.pi)**0.5)*np.exp(-(best - mu)**2 / (2*std**2))

    return ei

def plot(X, y, model):
    plt.scatter(X,y)
    Xsamples = np.array(np.arange(-5,5,0.01))
    Xsamples = Xsamples.reshape(len(Xsamples),1)
    ysamples, _ = surrogate_func(model, Xsamples)
    plt.plot(Xsamples, ysamples)
    plt.show()

def optimize_acquisition(X,y, model):

    Xsamples = np.random.uniform(-5,5, 100)
    Xsamples = Xsamples.reshape(len(Xsamples),1)

    scores = acquisition(X, Xsamples, model)
    ix = np.argmax(scores)
    return Xsamples[ix]

# X = np.random.uniform(-5,5,100)
# y = np.array([objective_func(x) for x in X])
X = np.array([0])
y = objective_func(X)
X = X.reshape(len(X),1)
y = y.reshape(len(y),1)

var_K = 0.3
# model = GaussianProcessRegressor()
model = GPR(var_K)
model.fit(X,y)

for i in range(100):
    # pdb.set_trace()
    x = optimize_acquisition(X,y,model)
    actual = objective_func(x)
    est, _ = surrogate_func(model, [[x]])
    print("x: {}, f(x): {}, actual: {}".format(x, est, actual))
    X = np.vstack((X, x.reshape(-1,1)))
    y = np.vstack((y,actual.reshape(-1,1)))
    
    model.fit(X,y)

plot(X,y,model)

ix = np.argmax(y)

print("Best result: x: {}, y: {}".format(X[ix], y[ix]))






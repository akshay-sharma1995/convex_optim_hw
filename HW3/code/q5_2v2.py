import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor

def objective_func(x):
    return x + np.sin(6*x)

def grad_f(x):
    return 1.0 + 6*np.cos(6*x)

def surrogate_func(model, X):
    return model.predict(X, return_std=True)

def acquisition(X, Xsamples, model):
    yhat, _ = surrogate_func(model, X)
    best = max(yhat)
    mu, std = surrogate_func(model, Xsamples)
    mu = mu[:,0]

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

    return Xsamples[ix, 0]

X = np.random.uniform(-5,5,100)
y = np.array([objective_func(x) for x in X])

X = X.reshape(len(X),1)
y = y.reshape(len(y),1)

model = GaussianProcessRegressor()

model.fit(X,y)

for i in range(100):
    x = optimize_acquisition(X,y,model)
    actual = objective_func(x)
    est, _ = surrogate_func(model, [[x]])
    print("x: {}, f(x): {}, actual: {}".format(x, est, actual))

    X = np.vstack((X, [[x]]))
    y = np.vstack((y,[[actual]]))
    
    model.fit(X,y)

plot(X,y,model)

ix = np.argmax(y)

print("Best result: x: {:.3f}, y: {:.3f}".format(X[ix], y[ix]))






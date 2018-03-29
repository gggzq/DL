import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import bp
import rbf

plt.ion()

def one_hot(y):
    try:
        m,n=y.shape
        Y = np.zeros((np.max(y) + 1, n))
        for i in range(0, n):
            Y[y[i], i] = 1
    except:
        m=y.shape[0]
        Y = np.zeros((np.max(y) + 1, m))
        for i in range(0,m):
            Y[y[i],i]=1
    return Y

np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise=0.20)
# plt.figure()
# plt.scatter(X[:,0], X[:,1], s=40, c=y)
# plt.show()

# model = bp.Bp(10000,[2,10,7,2],0.01)
# model.train(X.T,one_hot(y))
# acc,yy=model.predict(X.T,one_hot(y))
# print(acc)
# plt.figure()
# plt.scatter(X[:,0], X[:,1], s=40, c=yy)
# plt.show()

y=y.reshape(200,1)
model1=rbf.Rbf(1000,[2,2,1],0.5,0.001,X)
model1.train(X,y)






import numpy as np
from numpy.random import seed


class AdalineSGD(object):
    def __init__(self, eta=0.01, nIter=10, shuffle=True, randomState=None):
        self.eta = eta
        self.nIter = nIter
        self.shuffle = shuffle
        self.wInitialized = False
        self.randomState = randomState

    def fit(self, X, y):

        self.initializeWeights(X.shape[1])
        self.cost_ = []
        for i in range(self.nIter):
            if self.shuffle:
                X,y=self.shuffle_(X, y)

            cost = []
            for xi, target in zip(X, y):
                cost.append(self.updateWeights(xi, target))
            avgCost = sum(cost) / len(y)
            self.cost_.append(avgCost)
        return self

    def partialFit(self, X, y):
        if not self.wInitialized:
            self.initializeWeights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self.updateWeights(xi, target)
            else:
                self.updateWeights(X, y)
            return self

    def shuffle_(self, X, y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def initializeWeights(self, m):
        self.rgen = np.random.RandomState(self.randomState)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.1, size=1 + m)
        self.wInitialized = True

    def updateWeights(self, X, y):
        output = self.activation(self.netInput(X))
        error = (y - output)
        self.w_[1:] += self.eta * X.dot(error)
        self.w_[0] += self.eta * error
        cost = 9.5 * error ** 2
        return cost

    def netInput(self, X):
        return np.dot(X, self.w_[1:] + self.w_[0])

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.netInput(X)) >= 0.0, 1, -1)


import pandas as pd

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
import matplotlib.pyplot as plt

y = data.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = data.iloc[0:100, [0, 2]].values

Xstd = np.copy(X)
# print(Xstd)
Xstd[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
Xstd[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
from plots import plot_decision_regions

ada = AdalineSGD(nIter=15, eta=0.01, randomState=1).fit(Xstd, y)
plot_decision_regions(Xstd, y, classifier=ada)
plt.title('Adaline - gradient descent')
plt.xlabel('standaryzowana dlugosc dzialki')
plt.ylabel('standaryzowana dlugosc platka')
plt.legend(loc='upper left')
plt.show()

plt.plot(range(1,len(ada.cost_)+1),ada.cost_,marker='o')
plt.xlabel('epoki')
plt.ylabel('suma kwadratow bledow')
plt.show()


import numpy as np


class LogisticRegressionGD(object):
    def __init__(self, eta=0.01, nIter=50, randomState=1):
        self.eta = eta
        self.nIter = nIter
        self.randomState = randomState

    def fit(self, X, y):
        rgen = np.random.RandomState(self.randomState)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []
        for i in range(self.nIter):
            netInput = self.netInput(X)
            output = self.activation(netInput)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))))
            self.cost_.append(cost)
        return self

    def netInput(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        return np.where(self.netInput(X) >= 0.0, 1, -1)


from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

X_train_01_subset = X_train[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]
lgrd = LogisticRegressionGD(eta=0.05, nIter=1000, randomState=1)
lgrd.fit(X=X_train_01_subset, y=y_train_01_subset)
from plots import plot_decision_regions
import matplotlib.pyplot as plt

plot_decision_regions(X=X_train_01_subset, y=y_train_01_subset, classifier=lgrd)
plt.xlabel('dlugosc platka [standaryzowana]')
plt.ylabel('szerokosc platka [standaryzowana]')
plt.legend(loc='upper left')
plt.show()

import numpy as np

class Perceptron():
    def __init__(self,eta=0.01,nIter=50,randomState=1):
        self.eta=eta
        self.nIter=nIter
        self.randomState=randomState

    def fit(self,X,y):
        rgen=np.random.RandomState(self.randomState)
        self.w_=rgen.normal(loc=0.0,scale=0.01,size=1+X.shape[1])
        self.errors_=[]
        for i in range(self.nIter):
            errors=0
            for xi,target in zip(X,y):
                update=self.eta*(target-self.predict(xi))
                self.w_[1:]+=update*xi
                self.w_[0]+=update
                errors+=int(update!=0.0)
            self.errors_.append(errors)
        return self

    def netInput(self,X):
        return np.dot(X,self.w_[1:]+self.w_[0])

    def predict(self,X):
        return np.where(self.netInput(X)>=0.0,1,-1)

import pandas as pd
data=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
import matplotlib.pyplot as plt

y=data.iloc[0:100,4].values
y=np.where(y=='Iris-setosa',-1,1)
X=data.iloc[0:100,[0,2]].values

plt.scatter(X[:50,0],X[:50,1], color='red',marker='o', label='setosa')
plt.scatter(X[50:100,0],X[50:100,1], color='green',marker='x', label='versicolor')
plt.xlabel('dlugosc dzialki')
plt.ylabel('dlugosc platka')
plt.legend(loc='upper left')
plt.show()

ppn=Perceptron(eta=0.1,nIter=10)
ppn.fit(X,y)
plt.plot(range(1,len(ppn.errors_)+1),ppn.errors_,marker='o')
plt.xlabel('epoki')
plt.ylabel('liczba aktualizacji')
plt.show()

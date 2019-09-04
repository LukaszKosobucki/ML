import numpy as np

class AdalineGD(object):
    def __init__(self,eta=0.01,nIter=50,randomState=1):
        self.eta=eta
        self.nIter=nIter
        self.randomState=randomState

    def fit(self,X,y):
        rgen=np.random.RandomState(self.randomState)
        self.w_=rgen.normal(loc=0.0,scale=0.01,size=1+X.shape[1])
        self.cost_=[]
        for i in range(self.nIter):
            netInput=self.netInput(X)
            output=self.activation(netInput)
            errors=(y-output)
            self.w_[1:]+=self.eta*X.T.dot(errors)
            self.w_[0]+=self.eta*errors.sum()
            cost=(errors**2).sum()/2.0
            self.cost_.append(cost)
        return self

    def netInput(self,X):
        return np.dot(X,self.w_[1:]+self.w_[0])
    def activation(self,X):
        return X

    def predict(self,X):
        return np.where(self.netInput(X)>=0.0,1,-1)

import pandas as pd
data=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
import matplotlib.pyplot as plt

y=data.iloc[0:100,4].values
y=np.where(y=='Iris-setosa',-1,1)
X=data.iloc[0:100,[0,2]].values

"""
pokazane róznice wykresowo dla Adaline z krokiem uczenia 0,1 i 0.0001
wykresy pokazuja jak dobór kroku uczenia jest ważny
aby zbiegać do minimum lokalnego/globalnego 
a nie je przeskakiwac i zapetlac sie w bledzie
"""
# fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(10,4))
# ada1=AdalineGD(nIter=10,eta=0.01).fit(X,y)
# ax[0].plot(range(1,len(ada1.cost_)+1),np.log10(ada1.cost_),marker='o')
# ax[0].set_xlabel('epoki')
# ax[0].set_ylabel('log(suma kwadratow bledow)')
# ax[0].set_title('adaline - wspolczynnik uczenia 0,01')
# ada2=AdalineGD(nIter=50,eta=0.0001).fit(X,y)
# ax[1].plot(range(1,len(ada2.cost_)+1),np.log10(ada2.cost_),marker='o')
# ax[1].set_xlabel('epoki')
# ax[1].set_ylabel('suma kwadratow bledow')
# ax[1].set_title('adaline - wspolczynnik uczenia 0,0001')
#
# plt.show()


"""
użycie Adaline z wprowadzoną standaryzacją każdej j-tej cechy
pokazanie na wykresie podzielonych cech przez prostą
oraz sumy kwadratów błędów która dąży do 0
"""

Xstd=np.copy(X)
print(Xstd)
Xstd[:,0]=(X[:,0] - X[:,0].mean())/X[:,0].std()
Xstd[:,1]=(X[:,1] - X[:,1].mean())/X[:,1].std()
from plots import plot_decision_regions
ada=AdalineGD(nIter=15,eta=0.01).fit(Xstd,y)
plot_decision_regions(Xstd,y,classifier=ada)
plt.title('Adaline - gradient descent')
plt.xlabel('standaryzowana dlugosc dzialki')
plt.ylabel('standaryzowana dlugosc platka')
plt.legend(loc='upper left')
plt.show()

plt.plot(range(1,len(ada.cost_)+1),ada.cost_,marker='o')
plt.xlabel('epoki')
plt.ylabel('suma kwadratow bledow')
plt.show()

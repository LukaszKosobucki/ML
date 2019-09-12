from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
X_combined_std=np.vstack((X_train_std,X_test_std))
y_combined=np.hstack((y_train,y_test))
svm = SVC(kernel='linear',C=1.0,random_state=1)
svm.fit(X_train_std,y_train)
print(f"celnosc algorytmu: {np.round(svm.score(X_test_std,y_test)*100,2)}%")

from plots import plot_decision_regions
import matplotlib.pyplot as plt
plot_decision_regions(X_combined_std,y_combined,classifier=svm,test_idx=range(105,150))
plt.xlabel('dlugosc platka [standaryzowana]')
plt.ylabel('szerokosc platka [standaryzowana]')
plt.legend(loc='upper left')
plt.show()
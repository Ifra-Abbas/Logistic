from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

X,y=make_classification(n_samples=100,n_features=2,n_informative=1,n_redundant=0,
n_classes=2,random_state=41,n_clusters_per_class=1,class_sep=20,hypercube=False)

plt.figure(figsize=(10,6))
plt.scatter(X[:,0],X[:,1],c=y,cmap='winter',s=100)
# plt.show()
lor=LogisticRegression(penalty=None, solver="sag")
lor.fit(X,y)
print(lor.intercept_)
print(lor.coef_)
m1 = -(lor.coef_[0][0]/lor.coef_[0][1])
b1 = -(lor.intercept_/lor.coef_[0][1])
x_input = np.linspace(-3,3,100)
y_input = m1*x_input + b1

def gd(X,y):
    X=np.insert(X,0,1,axis=1)
    weights=np.ones(X.shape[1]) # no. of columns
    lr=0.5

    for i in range(5000):
        y_hat=sigmoid(np.dot(X,weights))
        weights=weights+lr*((np.dot((y-y_hat),X))/X.shape[0])
    return weights[0],weights[1:]
def sigmoid(z):
    return 1/(1+np.exp(-z))
intercept, coeff = gd(X,y)

m = -(coeff[0]/coeff[1])
b = -(intercept/coeff[1])
x_input1 = np.linspace(-3,3,100)
y_input1 = m*x_input1 + b
plt.figure(figsize=(10,6))
plt.plot(x_input,y_input,color='red',linewidth=3)
plt.plot(x_input1,y_input1,color='black',linewidth=3)
plt.scatter(X[:,0],X[:,1],c=y,cmap='winter',s=100)
plt.ylim(-3,2)
plt.show()
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt

X,y=make_classification(n_samples=100,n_features=2,n_informative=1,n_redundant=0,
n_classes=2,random_state=41,n_clusters_per_class=1,class_sep=10,hypercube=False)

plt.figure(figsize=(10,6))
plt.scatter(X[:,0],X[:,1],c=y,cmap='winter',s=100)
# plt.show()

def preceptron(X, y):
    X=np.insert(X,0,1,axis=1)
    wights=np.ones(X.shape[1]) # no. of columns 
    lr=0.1
    for i in range(1000):
        j=np.random.randint(1,100)
        y_hat=step(np.dot(X[j],wights))
        wights=wights+lr*(y[j]-y_hat)*X[j]
    return wights[0],wights[1:]
def step(z):
    if z>0:
        return 1
    else:
        return 0
intercept_,coeff=preceptron(X,y)
print(intercept_)
print(coeff)

m=-coeff[0]/coeff[1]
b=-intercept_/coeff[1]

x_input=np.linspace(-3,3,100)
y_input=m*x_input +b
plt.figure(figsize=(10,6))
plt.plot(x_input,y_input,color="red",linewidth=3)
plt.scatter(X[:,0],X[:,1],c=y,cmap='winter',s=100)
plt.ylim(-3,2)
plt.show()


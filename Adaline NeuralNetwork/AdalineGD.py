# Adaline; ADAptive LInear NEuron
import numpy as np
class AdalineGD(object):
    """ ADAptive LInear NEuron Classfier.
    Use gradient descent to find the right weights
    Parameters:
        eta: float, Learning rate between 0.0 and 1.0
        n_inter: iter, Number of iteration/passes over the training set
    
        w_: 1 dimesional array, weights after fitting
        errors_: list, number of misclassiciations in every epoch.
        """
    def __init__(self, eta = 0.01, n_iter = 50):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, Y):
        """ Fitting the training set with Gradient descent with identitcal Activation 
        funciton phi(W^T dot X)=W^T dot x
        X: Array-like Matrix, shape= n_samples x n_features,
        Training vectors where n_samples is the number of samples and 
        n_features is the number of features
        Y: array_like, shape = nsample
        Result of the training vectors.
        Returns self: object
        """
        self.w_ = np.zeros(1+X.shape[1])
        self.cost_ =[]

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (Y-output)
            self.w_[1:] +=self.eta*X.T.dot(errors)
            self.w_[0] += self.eta*errors.sum()
            cost = (errors**2).sum()/2.0
            self.cost_.append(cost)
        return self
            
    def net_input(self, X):
        """ Calculate Net input"""
        return np.dot(X, self.w_[1:])+self.w_[0]
        
    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)
    
    def predict(self, X):
        """ Return Predicted class label after Adaline"""
        return np.where(self.activation(X) >=0, 1, -1)

# Defining Printing graph function
from matplotlib.colors import ListedColormap
def plot_decision_regions(X,Y, classifier, resolution = 0.02 ):
    ''' set up marker geenraro and color map'''
    markers = ('s','o','x','^','v')
    colors = ('red','blue', 'lightgreen', 'gray' , 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(Y))])
    
    ''' Plotting the desicion region'''
    x1_min, x1_max = X[:,0].min()-1, X[:,0].max()+1
    x2_min, x2_max = X[:,1].min()-1, X[:,1].max()+1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min,x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha = 0.4, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(),xx2.max())
    
    '''plotting class samples'''
    for i, j in enumerate(np.unique(Y)):
        plt.scatter(x=X[Y==j,0], y=X[Y==j, 1],
                    alpha =0.8, c =cmap(i), marker = markers[i], label = j)
        
import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                 header=None)
Y = df.iloc[0:100,4].values
Y= np.where(Y == 'Iris-setosa', -1,1)
X= df.iloc[0:100,[0,2]].values

# Plot cost functions with eta =0.01 and 0.0001 against number of epochs

import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows = 1, ncols =2, figsize =(8,4))
ada1 = AdalineGD(n_iter = 10, eta = 0.01).fit(X,Y)
ax[0].plot(range(1,len(ada1.cost_)+1),np.log10(ada1.cost_),marker = 'o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')


ada2 = AdalineGD(n_iter = 10, eta = 0.0001).fit(X,Y)
ax[1].plot(range(1,len(ada2.cost_)+1),ada2.cost_,marker = 'o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')

# Featire scaling -- Standardization
X_std = np.copy(X)
X_std[:,0] = (X_std[:,0]-X_std[:,0].mean())/X_std[:,0].std()
X_std[:,1] = (X_std[:,1]-X_std[:,1].mean())/X_std[:,1].std()
ada = AdalineGD(n_iter = 15, eta =0.01).fit(X_std, Y)
plot_decision_regions(X_std, Y, classifier = ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('Sepal Length(standardized)')
plt.ylabel('Petal Length(standardized)')
plt.legend(loc = 'upper left')
plt.show()
plt.plot(range(1,len(ada.cost_)+1), ada.cost_, marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('Sum Squared Error')
plt.show()




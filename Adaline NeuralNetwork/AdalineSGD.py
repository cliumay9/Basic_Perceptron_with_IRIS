### AdalineSGD ###
# Adaline Stochastic Gradient Descent
from numpy.random import seed
import numpy as np

class AdalineSGD(object):
    """ADAptive LInear NEuron classifier.
    Use Stochastic gradient descent to find the right weights
    SGD update the weights incrementally for each training sample
    Parameters:
        eta: float, Learning rate between 0.0 and 1.0
        n_inter: iter, Number of iteration/passes over the training set
    
        w_: 1 dimesional array, weights after fitting
        errors_: list, number of misclassiciations in every epoch.
        shuffle: boolean, default: True, Shuffles training data every epoch
        if True to prevent cycle
        random_state: int, default: None, 
        Set random state for shuffling and initializing
        """
    def __init__(self, eta = 0.01, n_iter = 10,
                 shuffle = True, random_state = None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False #Status update on whether w_ is initialized
        self.shuffle = shuffle
        if random_state:
            seed(random_state)
            
    def fit(self,X , Y):
        """ Fitting the training set with Stochastic Gradient descent 
        with identitcal Activation 
        funciton phi(W^T dot X)=W^T dot x
        X: Array-like Matrix, shape= n_samples x n_features,
        Training vectors where n_samples is the number of samples and 
        n_features is the number of features
        Y: array_like, shape = nsample
        Result of the training vectors; its possible that Y is a matrix.
        Returns self: object
        """
        self._initialize_weights(X.shape[1])
        self.cost_ =[]
        for i in range(self.n_iter):
            if self.shuffle:
                X, Y = self._shuffle(X,Y)
            cost = []
            for xi, target in zip(X, Y):
                cost.append(self._update_weights(xi, target))
            avg_cost =sum(cost)/len(Y)
            self.cost_.append(avg_cost)
        return self
        
    def partial_fit(self, X, Y):
        """ Fit Training data without reinitializing the wiehgts.
        This partial fit funcition is for online training
        return self: object"""
        if not self.w_initialized: # If w_initialized is false
            self._initialize_weights(X.shape[1])
        print(Y.ravel().shape)
        if Y.ravel().shape[0]>1:
            
            # if there Y has more than one record so that it can be partial fit
            for xi, target in zip(X,Y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X,Y)
        return self
        
    def _shuffle(self, X, Y):
        """ Shuffle the training set; 
        so that it wont run into the cycle of optimizing
        """
        r = np.random.permutation(len(Y))
        return X[r], Y[r]

    def _initialize_weights(self, m):
        """ Initialize wiehgts to zeros"""
        self.w_ = np.zeros(1+m)
        self.w_initialized = True
    
    def _update_weights(self, xi, target):
        """ Using Adaline Learning rule (Stochastic Gradient Descent)
        to update the weihgts"""
        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.eta* xi.dot(error)
        self.w_[0] += self.eta*error
        cost = 0.5* error**2
        return cost
        
    def net_input(self, X):
        """Calculate net input """
        return np.dot(X, self.w_[1:])+ self.w_[0]

    def activation(self, X):
        """ Compute linear activation """
        return self.net_input(X)
        
    def predict(self, X):
        """ return predicted label after unit steps"""
        return np.where(self.activation(X)>=0.0, 1, -1)
                     
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

# Excuting AdalineSGD algorithm and visualize
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                 header=None)
Y = df.iloc[0:100,4].values
Y= np.where(Y == 'Iris-setosa', -1,1)
X= df.iloc[0:100,[0,2]].values
X_std = np.copy(X)
X_std[:,0] = (X_std[:,0]-X_std[:,0].mean())/X_std[:,0].std()
X_std[:,1] = (X_std[:,1]-X_std[:,1].mean())/X_std[:,1].std()

ada = AdalineSGD(n_iter = 15, eta =0.01, random_state=1)
ada.fit(X_std, Y)
plot_decision_regions(X_std, Y, classifier = ada)
plt.title('Adaline Stochastic Gradient Descent')
plt.xlabel('Sepal Length - std')
plt.ylabel('Petal Length - std')
plt.legend(loc = 'upper left')
plt.show()
plt.plot(range(1, len(ada.cost_)+1), ada.cost_, marker = 'o') 
plt.xlabel('Epochs')
plt.ylabel('Average cost') 
plt.show()      

# Then we can use ada.partial_fit(X_std[0,:], y[0]) 
# for online learning
        

        
        
        
        
        

                
        
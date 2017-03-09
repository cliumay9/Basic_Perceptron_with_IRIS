# Basic Perceptron   
      
import numpy as np

class Perceptron(object):
    """Perceptron classifier.
    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications (updates) in each epoch.
    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, Y):
        """Fit training data.
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        Returns
        -------
        self : object
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, Y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


# Convert csv to df
import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/''machine-learning-databases/iris/iris.data',
                 header = None)

df.tail()

# Visualize How our data look like
''' 
Extract first 100 samples; first 50 is Setosa while the second 50 is Versicolor
 convert class labels 1 for Versicolor and -1 for Setosa; 
 likely, extract the first feature column, 'Sepal length'
 third feature column 'petal length'
 '''

import matplotlib.pyplot as plt

# convert class labels forth feature column 1 for Versicolor (and virginica), if we extract first 150 isntead of 100) and -1 for Setosa; 
Y = df.iloc[0:100,4].values
Y = np.where(Y== 'Iris-setosa', -1,1)
X = df.iloc[0:150,[0,2]].values
plt.scatter(X[:50,0],X[:50,1],color = 'red', marker = 'o', label ='Setosa')
plt.scatter(X[50:100,0],X[50:100,1], color = 'blue', marker = 'x', label = 'Versicolor')
# plt.scatter(X[100:150,0],X[100:150,1],color = 'green', marker ='.', label = 'Virginica' )
plt.xlabel('Petal Length (cm)')
plt.ylabel('Sepal Length (cm)')
plt.legend(loc= 'upper left')
plt.show()

# Training our Perceptron on the iris dataset
pn = Perceptron(eta = 0.1, n_iter =10)
pn.fit(X,Y)

# Plot how many our models miscalssified of reach epoch in order to see whether an algorithm
# Converges
plt.plot(range(1, len(pn.errors_)+1), pn.errors_, marker ='o')
plt.xlabel('Epoch')
plt.ylabel('Number of Misclassficaiton')
plt.show()
# We found that our Perceptron converges after 6th epoch

'''Visualize the decison boundaries for the 2D datasets'''
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

plot_decision_regions(X,Y, pn)
plt.xlabel('Sepal length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(loc ='upper left')
plt.show()



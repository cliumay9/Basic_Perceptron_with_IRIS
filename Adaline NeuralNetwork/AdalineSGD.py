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
                X,Y = self.shuffle(X,Y)
            cost = []
            for xi, target in zip(X,Y):
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
        
        
        
        
        
        
        
        
        

                
        
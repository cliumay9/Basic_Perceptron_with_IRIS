# Basic Perceptron
'''Building a a class Perceptron and apply this model into the builtin Iris dataset'''

import numpy as np
class Perceptron(object):
    """ Perceptron classifier.
    Parameters:
        eta: float, the learning rate, bet. 0.0 and 1.0
        n_iter: int, number of iterations/pass over the training set
    Attributes:
        w_: 1 dimensional array, weights after fitting i.e. model weight
        errors_: list, number of misclassifications in every epoch.
        """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, Y):
        ''' Fitting traing data.
        Parameters: 
        X: array-like/Matrix, shape = [n_samples, n_features] n_samples x n_features
        Training vectors, n_samples = the number of samples, n_features = number of features/dimensions
        Y: array-like/(can be a matrix), shape =[n_samples]
            Predicted/target value
            
        returning self:object
        return
        self: object
            '''
        self.w_ = np.zeros(1+X.shape[1]) # Making a row of 0s
        self.erros_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,Y):
                update = self.eta*(target-self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update!=0.0)
            self.errors_.append(errors)
        return self
        
    def net_input(self,X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:])+self.w_[0]
        
    def predict(self, X):
        '''Return class label after unit steps'''
        return np.where(self.net_input(X)>=0.0, 1, -1)
            
import numpy as np
''' This standard scaler is utilized to scale data before entering to
a Machine Learning model '''

class Scaler:
    ''''Standard Scaler Class'''
    
    def fit(self, X):
        n = X.shape[0]               # No. dimensions
        self.m = np.mean(X, axis=1)  # maen of every variable
        self.s = np.std(X, axis=1)   # standard deviation for each variable
        self.n = n                   # dim
    
    def scale(self, X):              # scale the data 
        n = X.shape[0]               # new X dim
        assert n == self.n           # check the dim
        
        Xn = np.zeros(X.shape)
        for i in range(self.n):      # scale every single viariable 
            Xn[i, : ] = (X[i, : ] - self.m[i])/self.s[i] # normalize the data
        return Xn                    # retrun the scaled version of X
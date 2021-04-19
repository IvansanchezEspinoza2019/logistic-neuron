"""
    The matenathic architecture of the logistic neuron is a logistic function (1/(1+e^(-z))) 
    that belongs to the sigmoid functions family. This neuron is utulized to do both binary 
    clasifications {0,1} (a class) and also for doing probabilities regression [0,1].  
"""

import numpy as np

class LogisticNeuron:
    
    def __init__(self, n_dim, learn_fact):       # initialization
        self.w = -1 + 2 * np.random.rand(n_dim)  # synaptic weights vector
        self.b = -1 + 2 * np.random.rand()       # bias variable
        self.eta = learn_fact                    # neuron learning factor
        
    def predict_proba(self, X):                  # make probabilities predictions
        """
            This function predict continuous values [0,1]. This 
            is utilized to predict probabilities. 
        """
        z = np.dot(self.w, X) + self.b
        Y_predict = (1 / (1 + np.exp(-z)))       # activation function (logistic function)
        return Y_predict
    
    def predict(self, X, umbral=0.5):            # make discrete predictions
        """
            This function predict discrete values {0,1}. This 
            is utilized to make binary classifications.
        """
        z = np.dot(self.w, X) + self.b
        Y_predict = (1 / (1 + np.exp(-z)))       # activation function (logistic function)
        return 1 * (Y_predict > umbral)
        
    def _batcher(self, X, Y, batch_size):        # generating function that return all dataset but by parts (batch size)
        p = X.shape[1]
        li, ui = 0, batch_size
        while True:
            if li < p:
                yield X[:, li:ui], Y[:, li:ui]   # stops here, until its next call
                li, ui = li + batch_size, ui + batch_size
            else:
                return None
        
    def fit(self, X, Y, batch_size=1, epochs=100):      # train the neuron with mBGD (mini Batch Gradient Descent) algorithm 
        p = X.shape[1]                                  # patterns
        for _ in range(epochs):
            miniBatch = self._batcher(X, Y, batch_size) # call the generating function
            for mX, mY in miniBatch:                    # for every batch 
                Y_pred = self.predict_proba(mX)         # predict probabilities
                self.w += (self.eta/p) * np.dot((mY - Y_pred), mX.T).ravel() # adjust the synaptic weights
                self.b += (self.eta/p) * np.sum(mY - Y_pred)                 # adjust the bias variable
                
        
            

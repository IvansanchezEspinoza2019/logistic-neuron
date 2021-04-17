import numpy as np

class LogisticNeuron:
    def __init__(self, n_dim, learn_fact):       # initialization
        self.w = -1 + 2 * np.random.rand(n_dim)  # weights vector
        self.b = -1 + 2 * np.random.rand()       # bias variable
        self.eta = learn_fact                    # neuron learning factor
        
    def predict_proba(self, X):                  # make probabilities predictions
        """
            This function predict continuos values [0,1]. This 
            is utilized to predict probabilities. 
        """
        z = np.dot(self.w, X) + self.b
        Y_predict = (1 / (1 + np.exp(-z)))       # activation function (logistic function)
        return Y_predict
    
    def predict(self, X, umbral=0.5):            # make discret predictions
        """
            This function precdict discret values  {0,1}. This 
            is utilized to make binary clasifications.
        """
        z = np.dot(self.w, X) + self.b
        Y_predict = (1 / (1 + np.exp(-z)))       # activation function (logistic function)
        return 1 * (Y_predict > umbral)
        
    def _batcher(self, X, Y, batch_size):        # generating function that return all dataset but by parts (batch size)
        p = X.shape[1]
        li, ui = 0, batch_size
        while True:
            if li < p:
                yield X[:, li:ui], Y[:, li:ui]
                li, ui = li + batch_size, ui + batch_size
            else:
                return None
        
    def fit(self, X, Y, batch_size=1, epochs=100):      # train the neuron with mBGD (mini Batch Gradient Descent) algorithm 
        p = X.shape[1]                                  # patterns
        for _ in range(epochs):
            miniBatch = self._batcher(X, Y, batch_size) # call the gereration function
            for mX, mY in miniBatch:                    # for every mini batch
                Y_pred = self.predict_proba(mX)         # predict 
                self.w += (self.eta/p) * np.dot((mY - Y_pred), mX.T).ravel()
                self.b += (self.eta/p) * np.sum(mY - Y_pred)
                
        
            

"""
    In this little demo we make predictions over biologic cells dataset to determine if
    a set of cells are dangerous or not. We use the logistic neuron to make the classification
    (cancer cell or not cancer cell) but also can make probabilities predictions of
    being one on those classes.
"""

# libraries
from logistic_neuron import LogisticNeuron
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


#### LOAD CANCER DATASET ####
data = pd.read_csv('cancer.csv')

### 'X' WILL BE ALL INPUTS BUT NOT THE CLASS INPUT ###
x = np.asanyarray(data.iloc[:, :9]) 

### 'Y' TELL US TO WICH CLASS BELONGS EVERY SINGLE PATTERN (A CELL in X)  #####
y = np.asanyarray(data.iloc[:, 9]) #  class input

### SPLIT THE DATASET IN TRAIN AND TEST DATASETS ####
xtrain, xtest, ytrain, ytest = train_test_split(x, y)

#### CREATE OUR NEURON ####
neuron = LogisticNeuron(xtrain.T.shape[0], 0.1)

##### TRAIN THE NEURON #####
neuron.fit(xtrain.T, np.array([ytrain]), epochs=1000, batch_size=40)

#### MAKE PREDICTIONS #####
ypred = neuron.predict(xtest.T) 



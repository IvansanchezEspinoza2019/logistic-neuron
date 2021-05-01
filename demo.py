"""
    .: Demo for making some predictions with the logistic (sigmoid) neuron :. 

    In this demo I make predictions over two datasets:
        1: Cancer cells dataset
        2: People with Diabetes.
        
    In the first , we make predictions over biologic cells dataset to determine if
    a set of cells are dangerous or not. In the second one, we predict if people 
    have diabetes or not.
"""

####  libraries needed  #####
from logistic_neuron import LogisticNeuron
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score
from standard_scaler import Scaler
import numpy as np
import pandas as pd

opc = 1      # 1: cancer problem, 2: diabetes problem

if opc == 1:  ## CANCER DASTASET
    
    print("\t.: Cancer Dataset Selected :.")
    
    #### LOAD CANCER DATASET ####
    data = pd.read_csv('cancer.csv')

    ### 'X' WILL BE ALL INPUTS BUT NOT THE CLASS INPUT ###
    x = np.asanyarray(data.iloc[:, :9]) 

    ### 'Y' TELL US TO WICH CLASS BELONGS EVERY SINGLE PATTERN (A CELL in X)  #####
    y = np.asanyarray(data.iloc[:, 9]) #  class input

    ### SPLIT THE DATASET IN TRAIN AND TEST DATASETS ####
    xtrain, xtest, ytrain, ytest = train_test_split(x, y)

    #### DATASET DETAILS ###
    print("Total samples: ", x.shape[0])
    print("Samples for training: ", xtrain.shape[0])
    print("Samples for testing: ", xtest.shape[0])
    
    #### CREATE OUR NEURON ####
    neuron = LogisticNeuron(xtrain.shape[1], 0.1) #shape 1 is the problem dimension

    ### TO SCALE THE DATA BEFORE TRAINING THE NEURON  ####
    scaler = Scaler()
    
    ### SCALE BASED ON  MEAN AND STANDARD DEVIATION OF XTRAIN ###
    scaler.fit(xtrain.T)
    
    ### NORMALIZE XTRAIN ###
    normTrain = scaler.scale(xtrain.T)
    
    ##### TRAIN THE NEURON #####
    neuron.fit(normTrain, np.array([ytrain]), epochs=200, batch_size=32)
    
    ### ALSO SCALE XTEST DATASET ####
    normTest = scaler.scale(xtest.T)

    #### MAKE PREDICTIONS #####
    ytrainPred = neuron.predict(normTrain) # predict over the same samples with was trained before
    ytestPred = neuron.predict(normTest)   # predict new data the model does not saw

    ### PRINT SCORE ####
    print("\nPercentage Accuracy: ")
    print("\tTrain [%]: ", accuracy_score(ytrain, ytrainPred, normalize = True), "%")
    print("\tTest  [%]: ", accuracy_score(ytest, ytestPred, normalize = True), "%")
    
    print("\nCorrectly Predicted Samples: ")
    print("\tTrain [#]: ", accuracy_score(ytrain, ytrainPred, normalize = False)," of ",xtrain.shape[0])
    print("\tTest  [#]: ", accuracy_score(ytest, ytestPred, normalize = False), " of ", xtest.shape[0])
    
elif opc==2: ## DIABETES DATASET
    
    print(".: Diabetes Dataset Selected :.\n")
    ### DIABETES DATASET ####
    data = pd.read_csv("diabetes.csv")
    
    ### GET THE INPUTS ######
    x = np.asanyarray(data.iloc[:, :8])
    y = np.asanyarray(data.iloc[:, 8])
    
    ### SPLIT THE DATA ###
    xtrain, xtest, ytrain, ytest = train_test_split(x, y)
    
    ### DATASET DETAILS ###
    print("Total samples: ", x.shape[0])
    print("Samples for training: ", xtrain.shape[0])
    print("Samples for testing: ", xtest.shape[0])
    
    #### CREATE OUR NEURON ###
    neuron = LogisticNeuron(8, 0.1)
    
    ### TO SCALE THE XTRAIN AND XTEST DATA ####
    scaler = Scaler()
    
    #### THE SCALER LEARNED THE MEAN AND STANDARD DEVIATION OF THE TRAIN DATA ###
    scaler.fit(xtrain.T)
    
    ### NORMALIZE XTRAIN DATA BEFORE TRAINING THE NEURON ####
    normTrain = scaler.scale(xtrain.T)
    
    ### TRAIN THE NEURON WITH THE DATA NORMALIZED ####
    neuron.fit(normTrain, np.array([ytrain]), epochs=150,batch_size=32)
    
    #### ALSO WE HAVE TO NORMALIZE THE TEST DATASET BEFORE MAKING PREDICTIONS ####
    normTest = scaler.scale(xtest.T)
    
    #### MAKE PREDICTIONS ###
    ytrainPred = neuron.predict(normTrain)
    ytestPred = neuron.predict(normTest)
    
    #### RESULTS ####
    print("\nPercentage Accuraccy: ")
    print("Train [%]: ", accuracy_score(ytrain, ytrainPred,normalize =True),"%")
    print("Test  [%]: ", accuracy_score(ytest, ytestPred, normalize =True),"%")
    
    print("\nCorrectly Predicted Samples: ")
    print("Train [#]: ", accuracy_score(ytrain, ytrainPred,normalize = False), " of ",xtrain.shape[0])
    print("Test: [#]: ", accuracy_score(ytest, ytestPred, normalize = False), " of ", xtest.shape[0])

    
    
    
    
    

    
    
    
    

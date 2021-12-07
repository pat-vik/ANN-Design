#import the required libraries 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# read the preprocessed_data.csv file
data=pd.read_csv('preprocessed_data.csv')

'''
Design of a Neural Network from scratch

*************<IMP>*************
Mention hyperparameters used and describe functionality in detail in this space
- carries 1 mark

Hyperparameters: 
		
	The layers can be represented as [8,6,3,1]

	The two hidden layers use relu acivation function and the output layer uses sigmoid activation function.

	The loss function used is cross entropy.
	
	The optimization algorithm used is gradient descent.

	The learning rate is 0.001
	
	The number of epochs is 800
    
Functionality is explained in the README.txt file
'''
class NN:
    
    def __init__(self, layers=[8,6,3,1], l_rate=0.001, epochs=800):
    
        # initialize some of the hyperparameters
        self.layers = layers
        
        self.parameters = {}
        self.l_rate = l_rate
        self.epochs = epochs
        
        # Initialize the weights and biases from a random normal distribution
        np.random.seed(3)
        self.parameters['W1'] = np.random.randn(self.layers[0], self.layers[1])
        self.parameters['b1']  =np.random.randn(self.layers[1],)
        self.parameters['W2'] = np.random.randn(self.layers[1],self.layers[2])
        self.parameters['b2'] = np.random.randn(self.layers[2],)
        self.parameters['W3'] = np.random.randn(self.layers[2],self.layers[3])
        self.parameters['b3'] = np.random.randn(self.layers[3],)
        

    def relu(self,x):
        '''
        The relu function takes input as real numbers (x)
        output is 0 if x<=0
        output is x if x>0
        '''
        return np.maximum(0, x)
        
        
    def sigmoid(self,x):
        '''
        The sigmoid function takes input as real numbers and outputs a
        real-valued number between 0 and 1. 
        '''
        return 1.0 / (1.0 + np.exp(-x))
    
    def dRelu(self,x):
        # The derivative of relu is 1 if the input is greater than 1, and 0 otherwise
        x[x<=0] = 0
        x[x>0] = 1
        return x    

    ''' X and Y are dataframes '''

    def fit(self,X,Y):
        '''
        Function that trains the neural network by taking x_train and y_train samples as input
        '''
        self.X = X
        self.y = Y

        for i in range(self.epochs):
            yhat = self.forward_propagation()
            self.back_propagation(yhat)

                    
    def forward_propagation(self):
        '''
        Performs the forward propagation
        
        Z contains the intermediate calculation (Z=WX+b)
        W contains the weight
        b contains the bias
        A contains the activations for a layer 
        '''
        Z1 = self.X.dot(self.parameters['W1']) + self.parameters['b1']
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.parameters['W2']) + self.parameters['b2']
        A2 = self.relu(Z2)
        Z3 = A2.dot(self.parameters['W3']) + self.parameters['b3']
        yhat = self.sigmoid(Z3)

        # Store the values in the parameters dictionary so that it can be used during backpropagation
        self.parameters['Z1'] = Z1
        self.parameters['Z2'] = Z2
        self.parameters['Z3'] = Z3
        self.parameters['A1'] = A1
        self.parameters['A2'] = A2

        return yhat
    
    
    def back_propagation(self,yhat):
        '''
        Calculate the derivatives backward, from the loss all the way up to the first weight and bias
        The loss function used is cross entropy 
        '''
        
        # calculate the derivative of the loss with respect to the output yhat
        dl_yhat = -(np.divide(self.y,yhat) - np.divide((1 - self.y),(1-yhat)))        
        # calculate the derivative of sigmoid activation with respect to the loss
        dl_sig = yhat * (1-yhat)        
        # calculate the derivative of the loss wrt Z3
        dl_z3 = dl_yhat * dl_sig
        # calculate the derivative of the loss wrt A2
        dl_A2 = dl_z3.dot(self.parameters['W3'].T)
        # calculate the derivative of the loss wrt W3
        dl_w3 = self.parameters['A2'].T.dot(dl_z3)
        # calculate the derivative of the loss wrt b3
        dl_b3 = np.sum(dl_z3, axis=0)        
        # calculate the derivative of the loss wrt Z2
        dl_z2 = dl_A2 * self.dRelu(self.parameters['Z2'])
        # calculate the derivative of the loss wrt A1
        dl_A1 = dl_z2.dot(self.parameters['W2'].T)
        # calculate the derivative of the loss wrt W2
        dl_w2 = self.parameters['A1'].T.dot(dl_z2)
        # calculate the derivative of the loss wrt b2
        dl_b2 = np.sum(dl_z2, axis=0)
        # calculate the derivative of the loss wrt Z1
        dl_z1 = dl_A1 * self.dRelu(self.parameters['Z1'])
        # calculate the derivative of the loss wrt W1
        dl_w1 = self.X.T.dot(dl_z1)
        # calculate the derivative of the loss wrt b1
        dl_b1 = np.sum(dl_z1, axis=0)

        # update the weights and bias
        self.parameters['W1'] = self.parameters['W1'] - self.l_rate * dl_w1
        self.parameters['W2'] = self.parameters['W2'] - self.l_rate * dl_w2
        self.parameters['W3'] = self.parameters['W3'] - self.l_rate * dl_w3
        
        self.parameters['b1'] = self.parameters['b1'] - self.l_rate * dl_b1
        self.parameters['b2'] = self.parameters['b2'] - self.l_rate * dl_b2
        self.parameters['b3'] = self.parameters['b3'] - self.l_rate * dl_b3

    
    
    def predict(self,X):
        """
        The predict function performs a simple feed forward of weights
        and outputs yhat values 

        yhat is a list of the predicted value for df X
        """
        Z1 = X.dot(self.parameters['W1']) + self.parameters['b1']
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.parameters['W2']) + self.parameters['b2']
        A2 = self.relu(Z2)
        Z3 = A2.dot(self.parameters['W3']) + self.parameters['b3']
        prediction = self.sigmoid(Z3)
        yhat = prediction 
    
        return yhat
        
    
    def CM(self, y_test,y_test_obs):
        '''
        Prints confusion matrix 
        y_test is list of y values in the test dataset
        y_test_obs is list of y values predicted by the model

        '''

        for i in range(len(y_test_obs)):
            if(y_test_obs[i]>0.6):
                y_test_obs[i]=1
            else:
                y_test_obs[i]=0
        
        cm=[[0,0],[0,0]]
        fp=0
        fn=0
        tp=0
        tn=0
        
        for i in range(len(y_test)):
            if(y_test[i]==1 and y_test_obs[i]==1):
                tp=tp+1
            if(y_test[i]==0 and y_test_obs[i]==0):
                tn=tn+1
            if(y_test[i]==1 and y_test_obs[i]==0):
                fp=fp+1
            if(y_test[i]==0 and y_test_obs[i]==1):
                fn=fn+1
        cm[0][0]=tn
        cm[0][1]=fp
        cm[1][0]=fn
        cm[1][1]=tp

        p= tp/(tp+fp)
        r=tp/(tp+fn)
        f1=(2*p*r)/(p+r)
        accuracy=((tp+tn)/(tp+tn+fp+fn))*100
        
        print(f"Confusion Matrix : {cm}")
        print(f"Precision : {p}")
        print(f"Recall : {r}")
        print(f"F1 SCORE : {f1}")
        print(f"Accuracy : {accuracy}")
        
# Creating test and train split
X = data.drop(columns=['Result'])
y_label = data['Result'].values.reshape(X.shape[0], 1)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y_label, test_size=0.3, random_state=1)

# Standardize the data
sc = StandardScaler()
sc.fit(Xtrain)
Xtrain = sc.transform(Xtrain)
Xtest = sc.transform(Xtest)

# Create an object of the NN class and then train the model using the fit method 
nn=NN()
nn.fit(Xtrain, ytrain)
# Use the predict method to get predictions 
train_pred = nn.predict(Xtrain)
test_pred = nn.predict(Xtest)

# Print the metrics for the train and test data
print("Metrics for train data")
nn.CM(ytrain, train_pred)
print("\n")
print("Metrics for test data")
nn.CM(ytest, test_pred)

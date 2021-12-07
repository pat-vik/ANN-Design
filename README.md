# ANN-Design
This is an assignment submitted for the course Machine Intelligence(UE18CS303) offered by PES University.

Explanation of Implementation:
	
	The neural network consists of 4 layers in total that is 1 input layer, 2 hidden layers and 1 output layer. 
	The input layer contains 8 neurons. The next layer contains 6 neurons which is followed by a layer with 3 neurons and then the 
	output layer with 1 neuron.
  
	The "class NN" contains all the code to train and test the neural network. 
	When an object of the class NN is created the __init__ method gets called and this method initializes the hyperparameters of the 
	neural network. Weights and biases are randomly initialized from a normal distribution as numpy arrays and are stored in a 
	dictionary called "parameters".

	The fit method trains the neural network by taking x_train and y_train samples as input. This is acheived by looping through the
	forward propagation and back propagation methods for a specified number of epochs.

	The forward_propagation method calculates the intermediate Z values (Z=WX+b) and the activations for the layers and stores them 
	in the "parameters" dictionary. It uses relu activation function for both the hidden layers and sigmoid activation function for 
	the output layer. This method returns yhat which contains the activations for the output layer.
	Z contains the intermediate calculation (Z=WX+b)
        W contains the weight
        b contains the bias
        A contains the activations for a layer
	
	The back_propagation method calculates the derivatives backward, from the loss all the way up to the first weight and bias. 
	The loss function used is cross entropy.
	And then the weights and biases are updated using the learning rate.

	The predict method performs a simple feed forward of weights and outputs yhat values. yhat is a list of the predicted values for 
	dataframe X.

	The CM method is used to compute the confusion matrix and then the precision, recall, f1 score and accuracy.

	

Hyperparameters: 
		
	The layers can be represented as [8,6,3,1]

	The two hidden layers use relu acivation function and the output layer uses sigmoid activation function.

	The loss function used is cross entropy.
	
	The optimization algorithm used is gradient descent.

	The learning rate is 0.001
	
	The number of epochs is 800



Key features include:
	
	During preprocessing we observed that for all the missing values for 'Weight' column the 'Result' column had value 0.
	By just sorting the dataset according to the weights, we observed that for lower values of weights(30 to 35) the 'Result'
	was mostly 0. So we filled the missing values for the weights with 31 (randomly picked between 30 to 35). This helped the 
	neural network to learn the dataset much better.

	We used relu activation function in the hidden layers and then sigmoid function(which is best for binary classification) for 
	the output layer.

	We are able to acheive an accuracy of 94 for training data and an accuracy of 89 for the test data.


Beyond the basics:

	Being new to neural networks, i have mostly covered all the basics for the implementation.
	I have used the template provided and tried to achieve a descent accuracy.
	
	Standardization of the data is done. 
	Variables that are measured at different scales do not contribute equally to the model fitting & model learned function and might 
	end up creating a bias. Thus, to deal with this potential problem feature-wise standardization (μ=0, σ=1) is usually used prior to 
	model fitting.
	

Steps to run the file:
	
	The src folder contains main.py along with the preprocessed_data.csv. So all you need to do is run the main.py file and all metrics 
	related to the train and test data will be the output.

#Using keras to develop my first neural network using tutorial at https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

from keras.models import Sequential 
from keras.layers import Dense 
import numpy as np

#fix random seed for reproducibility 
np.random.seed(7)


# 1. load pima indians dataset
dataset = np.loadtxt("pima-indians-diabetes.data.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# 2. creating the model 
#Using rectifier (relu) activation function on first and hidden layers and sigmoid function (Specially to ensure output is 1 or 0)  on the output layer

# first layer has 12 neurons and expects 8 output layers
# Second hidden layers has 8 neurons 
# Output layer has 1 neuron (1/0) 

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))



# 3. Compile model 


#Necessary to specify loss function to evaluate a set of weights. Here we us
# we use logarithmic loss which for binary classfn is ninary_crossentroup
#GD algo used is Adam


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# 4.  Fit the model

# The number of iterations are seleected in epochs and nuber of instances th
# that are evualuated are selected in the batch_size 


model.fit(X, Y, epochs=1550, batch_size=15) #Better accuracy with high epoch


# 5. Evaluate the model 

#Important to evaluate inorder to check how well algo performs on new data
#Train data set using evaluate(0 and pass same input and op as used to train #model


scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calculate predictions
predictions = model.predict(X)

# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)


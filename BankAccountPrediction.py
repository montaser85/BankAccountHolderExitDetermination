import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  

# Importing the dataset
dataset = pd.read_csv('BankData.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

#Using Dummy Variable for removing the ordinality from the categorical value

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
#Getting rid of Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''
 the fit function uses the original form of data without any pre processing and also 
 find out the parameters for the 
 model which can generate the processed data. The transform function finds out the 
 processed data using the parameters which are generated from the fit function.
 The fit_trabsform function is the combination of both functions.
'''


# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
#Sequential module is used for initializing the neural network
from keras.models import Sequential
#Dense is used for creating layers in the ANN
from keras.layers import Dense
#For Dropout regularization
from keras.layers import Dropout

# Initialising the ANN (As sequence of layers)
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dropout(p = 0.1)) # p is the fraction of the nodes which will be disabled in each iteration
''' 
    1. Units is the number of nodes in the hidden layer. Specification of this number is an artistic 
    approach. There is no concrete rules for this. One safe trick is to take the average of the input node
    and output node.
    2. Kernel_initializer is for initializing the weights randomly. The uniform function initialize the
    weights according to uniform distribution and the values will be near to 0.
    3. Activation is the activation function implied in the Hidden Layer and for hidden layer ReLu or 
    rectifier activation function works best.
    4. input_dim is the number of nodes in the input layer(no of features in the training set)
'''
#Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.1))

'''
    For the second hidden layer the input layer node is need not to be specified. For the first hidden 
    layer it was specified because the hidden layer does not know what is the input layer and it's dimension.
    But, after creating the hidden layer the other subsequent hidden layers realize what is its input 
    layer.
'''
#Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
'''
    The suitable activation function for this geo demographic problem is sigmoid as because we will
    predict the probabilty of the binary outcome. If the output is categorical than "softmax" would be
    the precise activarion function 
'''

#Compiling the ANN

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics= ['accuracy'])
'''
    1. The optimizer parameter defines the algorithm which will be implied for adjusting the weights.
    Different SGD algorithms are there to do so. One of them is 'adam' which works really efficiently.
    2. The loss parameter represents the loss function which helps in updating the weights in the
    most efficient way. For binary output its value is 'binary_crossentropy' and for categorical output
    its value is 'categorical_entropy'
    3. The accuracy metrics is used for finding out the accuracy of the output which will help in 
    improving the accuracy of the model in the subsequent training steps.
'''

# Fitting the ANN to the training dataset
classifier.fit(X_train, y_train, batch_size=10, nb_epoch =100)
'''
    batch_size is the number of observations after which the weights will be updated and nb_epoch is the
    number of iterations which will be required for training the whole dataset.
'''

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results

# this predicts the probabilty of an observation being 1 or leaving the bank
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
''' 
    Convert the data into binary outcome format from probabilty, the range can be adjusted according to
    the sensitivity of the data
'''
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# Predicting a single new observation
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""

new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction= (new_prediction>0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)





# Part 4 - Evaluating, Improving and Tuning the ANN

# Evaluating the ANN (Doing K Fold Cross Validation)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense








def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, 
                         kernel_initializer = 'uniform', 
                         activation = 'relu', 
                         input_dim = 11))
    
    classifier.add(Dense(units = 6, 
                         kernel_initializer = 'uniform', 
                         activation = 'relu'))
    classifier.add(Dense(units = 1, 
                         kernel_initializer = 'uniform', 
                         activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam',
                       loss = 'binary_crossentropy', 
                       metrics = ['accuracy'])
    return classifier









classifier = KerasClassifier(build_fn = build_classifier,
                             batch_size = 10, epochs = 100)





accuracies = cross_val_score(estimator = classifier, 
                             X = X_train, y = y_train, 
                             cv = 10, n_jobs = 1)

'''
    CV is the value of K (K Fold Cross Validation), n_jobs is the number of CPUs assigned for the 
    calculation. n_jobs=-1 means all the CPUs.
'''
mean = accuracies.mean()
variance = accuracies.std()




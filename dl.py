#%%
from cmath import exp
import os
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
# %%
df = pd.read_csv('insurance.csv')
# %%
X_train, X_test, y_train, y_test = train_test_split(df[['age', 'affordibility']], df['bought_insurance'], test_size=0.2, random_state=15)
#
# Scaling values to have both the variables in the same range of 0 to 1. 
X_train['age'] = X_train['age']/100
X_test['age'] = X_test['age']/100
# %%
model = keras.Sequential([
    keras.layers.Dense(1, input_shape = (2,), activation = 'sigmoid', kernel_initializer  = 'ones', bias_initializer = 'zeros')
])

model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy']
)

model.fit(X_train, y_train, epochs = 100)
#%%
model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)
#%%
coef, intercept = model.get_weights()
print(coef)
print(intercept)
# %%
# retrieving weights and bias from scratch. -  Manual Method. 

# 'activation' variable in the tensorflow model. 
def sigmoid(x):
    import math
    return 1/(1+math.exp(x))

# Prediction funciton takes in w1, w2, and the bias(intercept) and tells whether a person will buy insurance or not. 
def prediction_function(age, affordibility):
   weighted_sum =  coef[0]*age+coef[1]*affordibility + intercept
   return sigmoid(weighted_sum)

# Will convert vlaues in numpy array to 0/1 using sigmoid. 
def sigmoid_numpy(x):
   return 1/(1+np.exp(-x))

def log_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred_new = [max(epsilon, i) for i in y_pred]
    y_pred_new = [min(1-epsilon, i) for i in y_pred_new]
    y_pred_new = np.array(y_pred_new)
    return -np.mean(y_true*np.log(y_pred_new)+(1-y_true)*np.log(1-y_pred_new))


#%%

class myNN:
    def __init__(self):
        self.w1 = 1
        self.w2 = 1
        self.bias = 0
        
        
    def gradient_descent(self, age, affordibility, y_true, epochs, loss_thresold):
        w1 = w2 = 1
        bias = 0
        rate = 0.5
        n = len(age)
        for i in range(epochs):
            # Standard formula for one neuron until here. (weight*input + bias.)
            weighted_sum = w1 * age + w2 * affordibility + bias
            y_predicted = sigmoid_numpy(weighted_sum)
            loss = log_loss(y_true, y_predicted)
            
            # np.dot is for matrix multiplication. 
            w1d = (1/n)*np.dot(np.transpose(age),(y_predicted-y_true)) 
            w2d = (1/n)*np.dot(np.transpose(affordibility),(y_predicted-y_true)) 

            bias_d = np.mean(y_predicted-y_true)
            w1 = w1 - rate * w1d
            w2 = w2 - rate * w2d
            bias = bias - rate * bias_d
            
            if i%50==0:
                print (f'Epoch:{i}, w1:{w1}, w2:{w2}, bias:{bias}, loss:{loss}')
            
            if loss<=loss_thresold:
                print (f'Epoch:{i}, w1:{w1}, w2:{w2}, bias:{bias}, loss:{loss}')
                break

        return w1, w2, bias    
        
    def fit(self, X, y, epochs, loss_thresold):
        self.w1, self.w2, self.bias = self.gradient_descent(X['age'],X['affordibility'],y, epochs, loss_thresold)
        print(f"Final weights and bias: w1: {self.w1}, w2: {self.w2}, bias: {self.bias}")
        
    def predict(self, X_test):
        weighted_sum = self.w1*X_test['age'] + self.w2*X_test['affordibility'] + self.bias
        return sigmoid_numpy(weighted_sum)


# %%
# Creating class object to test out:
customModel = myNN()
# %%
customModel.fit(X_train, y_train, epochs= 1000, loss_thresold=0.50 )
# %%
#------------------------------------------------------------------------------------
# Stochastic gradient descent(SGD) and Batch gradient descent(BGD). 
#
# SGD uses one training sample (variable) for one epoch and then adjusts weight. - Good for really big data sets. Lots of computational power. 
#
# BGD uses all training samples for one epoch and then adjusts weight.  - Good for smaller data sets. 

home = pd.read_csv('homeprices.csv')
from sklearn.preprocessing import MinMaxScaler
scale  =  MinMaxScaler()
#sx = sx.scale.fit_transform()
# %%
X_scaled = scale.fit_transform(home.drop('price', axis = 1))
# Reshaping the y data set into 20 rows and one column. Converting into (20,1) matrix (1D) -> (2D). 
y_scaled = scale.fit_transform(home['price'].values.reshape(home.shape[0],1))

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size = 0.2, random_state= 30)

# %%
# BGD
def batch_gradient_descent(X, y_true, epochs, learning_rate = 0.01):
    number_of_features = X.shape[1]
    w = np.ones(shape = number_of_features)
    b = 0
    total_samples = X.shape[0]
    costlist = []
    epochlist = []
    
    for i in range(epochs):
        # weight * input + bias
        y_predicted = np.dot(w, X.T) + b 

        # derivative 
        w_grad = -(2/total_samples)*(X.T.dot(y_true-y_predicted))
        b_grad = -(2/total_samples)*np.sum(y_true-y_predicted)
        
        w = w - learning_rate * w_grad
        b = b - learning_rate * b_grad
        
        cost = np.mean(np.square(y_true- y_predicted)) # MSE - mean squared error
    
        if i%10 == 0:
            costlist.append(cost)
            epochlist.append(i)
            
    return w, b, cost, costlist, epochlist
#%%
import matplotlib.pyplot as plt

plt.xlabel('epochs')
plt.ylabel('cost')
plt.plot(epochlist, costlist)
# %%


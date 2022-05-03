#%%
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




# %%

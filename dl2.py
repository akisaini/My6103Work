#%%
import numpy as np
from sympy import Min
import tensorflow as tf
from tensorflow import keras 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# %%
y_predicted = np.array([1,1,0,0,1])
y_true = np.array([0.30, 0.7, 1, 0, 0.5])
# %%
def mae(y_true, y_predicted):
    total = 0
    for y_t, y_p  in zip(y_true, y_predicted):
        total += abs(y_t - y_p) 
    return print(f'MAE:{(total)/len(y_true)}')
# %%
def mse(y_true, y_predicted):
    total = 0
    for y_t, y_p  in zip(y_true, y_predicted):
        total += (y_t - y_p)
    return print(f'MSE:{(total)**2/len(y_true)}')
# %%
def log_loss(y_true, y_predicted):
    epsilon = 1e-15 # 0.000000000000001
    # replacing zero with epsilon -> very close to zero.
    y_predicted_new = [max(i, epsilon) for i in y_predicted]
    # replacing 1 with 1-epsilon -> very close to one. 
    y_predicted_new = [min(i, 1-epsilon) for i in y_predicted_new]
    y_predicted_new = np.array(y_predicted_new)
    return print(f'Log Loss Value: {-np.mean(y_true*np.log(y_predicted_new)+(1-y_true)*np.log(1-y_predicted_new))}')

# %%
df = pd.read_csv('insurance.csv')
# %%
# Gradient Descent 
# iloc[rows, columns]
X = df.iloc[:,:2]
y = df.iloc[:,2]

X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size = 0.2, train_size = 0.8, shuffle  = True, random_state= 20)

# %%
# scaling the data to bring age and affordability on the same scale. 

X_train_scaled = X_train.copy()
X_train_scaled['age'] = X_train_scaled['age']/100 
# %%
X_test_scaled = X_test.copy()
X_test_scaled['age'] = X_test_scaled['age']/100 
# %%
 
# building model with one layer.

model = keras.Sequential([keras.layers.Dense(1, input_shape = (2,), activation = 'sigmoid', kernel_initializer = 'ones', bias_initializer = 'zeros')])

model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = 1000)
# %%
# creating neural network class: 
def sigmoid_numpy(x):
   return 1/(1+np.exp(-x))
#%%
class basicNN:
    
    def __init__(self):
        self.w1 = 1
        self.w2 = 1
        self.bias = 0
        
    def gradient_descent(self, x1, x2, y_true, epochs, loss_threshold):
        w1 = w2 = 1
        bias = 0
        rate = 0.5 # this is kept constant. 
        n = len(x1)
        for i in range(epochs):
            # Standard formula for one neuron until here. (weight*input + bias.)
            weighted_sum = w1 * x1 + w2 * x2 + bias
            y_predicted = sigmoid_numpy(weighted_sum)
            # calculating loss now
            loss = log_loss(y_true, y_predicted)
            
            # calculating partial derivatives
            # np.dot is for matrix multiplication. 
            w1d = (1/n)*np.dot(np.transpose(x1),(y_predicted-y_true)) 
            w2d = (1/n)*np.dot(np.transpose(x2),(y_predicted-y_true)) 

            bias_d = np.mean(y_predicted-y_true)
            w1 = w1 - rate * w1d
            w2 = w2 - rate * w2d
            bias = bias - rate * bias_d
            
            if i%50==0:
                print (f'Epoch:{i}, w1:{w1}, w2:{w2}, bias:{bias}, loss:{loss}')
            
            if loss<=loss_threshold:
                print (f'Epoch:{i}, w1:{w1}, w2:{w2}, bias:{bias}, loss:{loss}')
                break

        return w1, w2, bias
    
    
    def fit(self, X, y, epochs, loss_threshold):
        self.w1, self.w2, self.bias = self.gradient_descent(X['age'], X['affordibility'], y, epochs, loss_threshold)
        print(f'final weights and bias are: {self.w1}, {self.w2}, {self.bias}')  
        
    def predict(self, X_test):
        weighted_sum = self.w1*X_test['age'] + self.w2*X_test['affordability'] + self.bias
        return sigmoid_numpy(weighted_sum)
#%%
my_model = basicNN()
# %%
my_model.fit(X_train_scaled, y_train, epochs = 100, loss_threshold = 0.20)

# %%
# Customer churn prediction using ANN. 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
#%%
df = pd.read_csv('TelcoCustomerChurn.csv')
# %%
df
# %%
df.drop('customerID', axis = 1, inplace = True)
# %%
df.replace({'No': 0, 'Yes':1, 'No internet service':0}, inplace = True)
df.replace({'Male': 1, 'Female':0 }, inplace = True)
df.replace({'No phone service': 0 }, inplace = True)
df = pd.get_dummies(df, columns = ['InternetService', 'Contract', 'PaymentMethod' ])
df = df[df['TotalCharges']!=' ']
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
# %%
# Now the data is ready. Lets scale it. 
cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
scale = MinMaxScaler()
for i in cols_to_scale:
    df[cols_to_scale] = scale.fit_transform(df[cols_to_scale])
# %%
from sklearn.model_selection import train_test_split
X = df.drop('Churn', axis = 1)
y = df['Churn']
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 0.8, test_size= 0.2, random_state= 10)
# %%
# Now we are ready for building neural network (ANN):
from tensorflow import keras
import tensorflow as tf
# %%
model = keras.Sequential([
                        keras.layers.Dense(26, input_shape= (26,), activation = 'relu'), # input layer
                        keras.layers.Dense(15, activation = 'relu'), # hidden layer (can have any number of neurons)
                        keras.layers.Dense(10, activation = 'relu'), # another hidden layer
                        keras.layers.Dense(1, activation = 'sigmoid') # output layer -  sigmoid because target is 1/0
                        ])

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = 500)
# %%
y_pred = model.predict(X_test)
# converting all numbers above 0.5 to 1 and below 0.5 to 0.
y_p = []
for i in y_pred:
    if i[0] > 0.5:
        y_p.append(1)
    else:
        y_p.append(0)
# %%
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_p)
# %%
cm
# %%

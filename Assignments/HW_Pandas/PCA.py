#%%
from os import sep
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#%%
df = pd.read_csv('heart.csv', sep=',')
#no null values. 
# %%

# %%
df.Sex.unique()
# %%
# converting factor variables to numeric values. 
df['Sex'].replace({
    'M':1,
    'F':0
}, inplace= True)
df['ChestPainType'].replace({
    'ATA':0,
    'NAP':1,
    'ASY':2,
    'TA':3
}, inplace = True)
df['RestingECG'].replace({
    'Normal':0,
    'ST':1,
    'LVH':2,
}, inplace = True)
df['ExerciseAngina'].replace({
    'N':0,
    'Y':1,
}, inplace = True)
df['ST_Slope'].replace({
    'Up':0,
    'Flat':1,
    'Down':2,
}, inplace = True)

# %%
X = df.drop(['HeartDisease'], axis = 1)
# %%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, df.HeartDisease, test_size = 0.3, random_state= 10)

# %%
from sklearn.svm import SVC

model1 = SVC()
model1.fit(X_train,y_train)

# %%
model1.score(X_test, y_test)
# %%
from sklearn.linear_model import LogisticRegression

model2 = LogisticRegression()
model2.fit(X_train, y_train)

# %%
model2.score(X_test, y_test)
# %%
from sklearn.ensemble import RandomForestClassifier
model3 = RandomForestClassifier()
model3.fit(X_train, y_train)
# %%
model3.score(X_test, y_test)
# %%
# Lets say we go ahead with SVM. 
# %%
from sklearn.decomposition import PCA
pca = PCA(n_components= 6) # chooses 6 of the most important columns/variables in the dataset. 

x_pca = pca.fit_transform(X)
# %%
X_train_pca, X_test_pca, y_train, y_test = train_test_split(x_pca, df.HeartDisease, test_size=0.3, random_state=10)
# %%
model1.fit(X_train_pca, y_train)
# %%
model1.score(X_test_pca, y_test) # ~71% accuracy with 6 components.
# %%
model3.fit(X_train_pca, y_train)
# %%
model3.score(X_test_pca, y_test) # ~76% accuracy with 6 components. 
# %%
y_pred_pca = model3.predict(X_test_pca)
# %%
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_pca)
cm
# %%
import seaborn as sn
sn.heatmap(cm, annot  = True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
# %%
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_pca))
# 77% accuracy of the model. 
# %%

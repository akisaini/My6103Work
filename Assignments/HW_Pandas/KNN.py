#%%
from cProfile import label
from sklearn.datasets import load_digits
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
# %%
digits = load_digits()
 
df = pd.DataFrame(digits.data)
df['target'] = digits.target
# %%
dfgb = df.groupby('target')
# %%
df_0 = dfgb.get_group(0)
df_1 = dfgb.get_group(1)
df_2 = dfgb.get_group(2)
df_3 = dfgb.get_group(3)
df_4 = dfgb.get_group(4)
df_5 = dfgb.get_group(5)
df_6 = dfgb.get_group(6)
df_7 = dfgb.get_group(7)
df_8 = dfgb.get_group(8)
df_9 = dfgb.get_group(9)    
    
# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test  = train_test_split(df.drop(['target'], axis = 1) , df['target'], test_size = 0.3, random_state= 10)
# %%
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors = 5)
model.fit(X_train, y_train)
# %%
model.score(X_test, y_test)
# %%
from sklearn.metrics import confusion_matrix
y_pred = model.predict(X_test) # predicting values for X_test -> y_pred
cm = confusion_matrix(y_test, y_pred) # comparing original values with predicted values using the confusion_matrix. 
cm # 
# %%
sn.heatmap(cm, annot=True)
plt.xlabel('Truth')
plt.ylabel('Predicted')
plt.show()
# %%

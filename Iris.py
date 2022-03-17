#%%
from numpy import column_stack
from sklearn import datasets 
import matplotlib.pyplot as plt 

digits = datasets.load_digits()

#%%
for i in range(0,5):
    plt.matshow(digits.images[i])
#%%
dir(digits)

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

# %%
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)
# %%
model.fit(x_train, y_train)
# %%
model.fit(x_test, y_test)
# %%
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
# %%
dir(iris)
df = pd.DataFrame(iris.data, columns = iris.feature_names)
df
# %%
plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'])
plt.show()
plt.scatter(df['petal length (cm)'], df['petal width (cm)'], marker ='*')
plt.show()
# %%
df['target'] = iris.target
df.tail(10)
# %%
df0 = df[df['target'] == 0]
df1 = df[df['target'] == 1]
df2 = df[df['target'] == 2]
# %%
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'], color= 'blue')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], color= 'green')
plt.scatter(df2['sepal length (cm)'], df2['sepal width (cm)'], color= 'red')
plt.show()


plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'], color= 'blue')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'], color= 'green')
plt.scatter(df2['petal length (cm)'], df2['petal width (cm)'], color= 'red')
plt.show()

# %%
X = df.drop(['target'], axis = 1)
X.head()
# %%
y = df['target']
y
# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# %%
from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)
# %%
model.score(X_test, y_test)
# %%

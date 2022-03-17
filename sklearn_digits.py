#%%
from sklearn.datasets import load_digits
import pandas as pd
import matplotlib.pyplot as plt
# %%
digits = load_digits()
dir(digits)
# %%
for i in range(4):
    plt.matshow(digits.images[i])
    plt.show()
# %%
df = pd.DataFrame(digits.data)
# %%
df['target'] = digits.target
y = df['target']
X = df.drop(['target'], axis = 1)
# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# %%
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)
# %%
model.score(X_test, y_test)
# %%
y_predict = model.predict(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_predict)
# %%
import seaborn as sn
sn.heatmap(cm, annot=True)
# %%

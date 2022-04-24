#%%[markdown]
#
# # HW8 - Classifiers on Digits dataset
# 
# A more comprehensive dataset on digits is available (on Blackboard). It is quite large, 
# with 60k observations. Each oberservatin is a 28x28 pixel gray scale image (digitized), 
# and 256 gray levels. The first column of the dataset (csv) is the target y value (0 to 9). 
# The remaining 784 columns are the gray values for each pixel for each observation.
#  
# ## Question 1: Read in the dataset
# First, unzip the data file. 
# There is no column names in the csv. You better create a list of names like x0, x1, 
# or x000, x001, etc, for the 784 columns plus the y-target. Use the list when creating 
# the dataframe. 
# Which column is the y-target?
#
# Check the shape and data type, make sure everything looks fine.
#
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

df = pd.read_csv('mnist_train.csv', header = None)
dict = {df.columns[i+1]: 'x'+str(i) for i in range(784)}
df = df.rename(columns = dict)
df.columns.values[0] = 'y_target'

df.head()
print(df.shape) # (60000, 785)
# The 'df.columns.values[0]' is the target variable column. 

#%%
# ## Question 2: Preparing the data
# On my system, if I use all 60k observations, it took a long time to run the classifiers. 
# I ended up retaining only 8k observations. They are already randomized. The first 
# 8k rows work for me. If you system is faster/slower, you should adjust the total 
# accordingly. Keep in mind however you better have many more rows than columns.
# 
# Chossing the first 8k rows from the df set. 

df = df.iloc[:8000,:]
df.shape
#%%
# Now prepare for the 4:1 train-test split.
# 
X = df.drop('y_target', axis = 1) # regressor set
y = df['y_target'] # target set
#%%
# If the latter modeling part does not run, check the X_train, X_test has the 
# right object type. Use the 8x8 pixel sample in class as a guide. 
# 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state= 15, stratify=y)

# train set size: 6000 rows
# test set size: 2000 rows
#%% 
# ## Question 3: View some samples 
# Plot the first and the last row of your train set, and see the image as we 
# did in class. Make sure the format is a 28x28 array for the plot to work.
# 
plt.gray()
plt.matshow(X_train.iloc[0:1,:]) # first row
plt.show()
plt.matshow(X_train.iloc[5999:6000, :]) # last row
#%%
# ## Question 4: Run the six classifiers
# For each each, print the train score, the test score, the confusion matrix, and the classification report.
# 
#%%
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#%%
# * SVC(): you can try adjusting the gamma level between 'auto', 'scale', 0.1, 5, etc, and see if it makes any difference 
# * SVC(kernel="linear"): having a linear kernel should be the same as the next one, but the different implementation usually gives different results 
model_svc = SVC()
model_svc.fit(X_train, y_train)
print(model_svc.score(X_test, y_test))
y_pred_svc = model_svc.predict(X_test)
svc_cm = confusion_matrix(y_test, y_pred_svc)
claf_rep_svm = classification_report(y_test, y_pred_svc)
print(claf_rep_svm)
# At gamma 'auto' model score was 96%. On average f1 score was 96% as well. 
#%%
# At gamma = 2
model_svc2 = SVC(gamma = 2)
model_svc2.fit(X_train, y_train)
print(model_svc2.score(X_test, y_test))
y_pred_svc2 = model_svc2.predict(X_test)
svc_cm2 = confusion_matrix(y_test, y_pred_svc2)
claf_rep_svm2 = classification_report(y_test, y_pred_svc2)
print(claf_rep_svm2)
#%%
# At gamma = 5
model_svc3 = SVC(gamma = 5)
model_svc3.fit(X_train, y_train)
print(model_svc3.score(X_test, y_test))
y_pred_svc3 = model_svc3.predict(X_test)
svc_cm3 = confusion_matrix(y_test, y_pred_svc3)
claf_rep_svm3 = classification_report(y_test, y_pred_svc3)
print(claf_rep_svm3)

# As the value of gamma increases the model is becoming more overfit. The model overfit the training set as the gamma value increases. 

#%%
# * LinearSVC() 

model_lsvc = LinearSVC(max_iter = 1000)
model_lsvc.fit(X_train, y_train)
print(model_lsvc.score(X_test, y_test))
y_pred_lsvc = model_lsvc.predict(X_test)
lsvc_cm = confusion_matrix(y_test, y_pred_lsvc)
claf_rep_lsvc = classification_report(y_test, y_pred_lsvc)
print(claf_rep_lsvc)

# Model produced 85.75% Accuracy.
#%%
# * LogisticRegression()

model_logit = LogisticRegression(max_iter = 10000)
model_logit.fit(X_train, y_train)
print(model_logit.score(X_test, y_test))
y_pred_logit = model_logit.predict(X_test)
logit_cm = confusion_matrix(y_test, y_pred_logit)
claf_rep_logit = classification_report(y_test, y_pred_logit)
print(claf_rep_logit)
# Model produced 88% Accuracy 
#%%

# * KNeighborsClassifier(): you can try different k values and find a comfortable chknn

model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_train, y_train)
print(model_knn.score(X_test, y_test))
y_pred_knn = model_knn.predict(X_test)
knn_cm = confusion_matrix(y_test, y_pred_knn)
claf_rep_knn = classification_report(y_test, y_pred_knn)
print(claf_rep_knn)
# KNN produced a 94% accuracy in the model with K = 3. With lower values of k the model will undergo overfitting. 
#%%
# * DecisionTreeClassifier(): try 'gini', 'entropy', and various max_depth  
model_dtc = DecisionTreeClassifier(max_depth = 7, random_state=1, criterion='gini')
model_dtc.fit(X_train, y_train)
print(model_dtc.score(X_test, y_test))
y_pred_dtc = model_dtc.predict(X_test)
dtc_cm = confusion_matrix(y_test, y_pred_dtc)
claf_rep_dtc = classification_report(y_test, y_pred_dtc)
print(claf_rep_dtc)
# Model produced 51% accuracy with max_depth = 3, random_state as true and criterion set to 'gini'
 
# Model produced 54% accuracy with max_depth = 3, random_state as true and criterion set to 'entropy'

# As depth is set tp lower value, the accuracy of the model decreases. If set to a high value, the accuracy of the model increases. With depth set to 7, model accuracy is 77%. This suggests that increasing depth probably leads to overfitting in the model. 
#%%
#  
#  
# ## Question 5: Cross-validation 
from sklearn.model_selection import cross_val_score
# Use cross-validation to get the cv scores (set cv=10, and use the accuracy score) for the six classifiers. 
#%%
# You can use the X_test and y_test for that instead of the one we picked out. You might or might not have 
# that complete set separated into X and y, although it should be easy.
# 
# When you use cross validation, it will be a few times slower than before as it score each model 10 different times.
# 
# While we are at it, let us try to time it. If the you the magic python functions (%), 
# you can easily clock the executation time of a line of code. Instad of this:    
# 
# tree_cv_acc = cross_val_score(tree, X_cv, y_cv, cv= 10, scoring="accuracy") 
# OR  
# tree_cv_acc = cross_val_score(tree, X_cv, y_cv, cv= 10, scoring="accuracy", n_jobs = -1) 
# n_jobs = -1 will use all the core/CPU in your computer. Notice the difference in speed.  
# https://ipython.readthedocs.io/en/stable/interactive/magics.html?highlight=%25time#magic-time
# 
# we use this:     
# 
# %timeit tree_cv_acc = cross_val_score(tree, X_train, y_train, cv= 10, scoring='accuracy')    
# And I get, without n_jobs: ~ 18.2 s ± 167 ms per loop (mean ± std. dev. of 7 runs, 1 loop each) 
# With n_jobs=-1, I have ~ 3.18 s ± 277 ms per loop (mean ± std. dev. of 7 runs, 1 loop each) 
# These are with 20k data rows.
# 
# Note that %timeit will also try a few (default 7) runs to find out the average time of 
# the cv function took.  The tree algorithm turns out not too bad.

model_svc_cv_acc = cross_val_score(model_svc, X_train, y_train, cv= 10, scoring='accuracy')
# 22.4 ns ± 0.644 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each) - Statement overall took 4m 44.3s
model_svc_cv_acc2 = cross_val_score(model_svc, X_train, y_train, cv= 10, scoring='accuracy', n_jobs = -1)
# 23.8 ns ± 0.755 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)
# Statement overall took 4m 13.8s 
model_logit_cv_acc = cross_val_score(model_logit, X_train, y_train, cv= 10, scoring='accuracy')
# 25.1 ns ± 2.06 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)
# Statement overall tool 9m 6.9s 
model_knn_cv_acc = cross_val_score(model_knn, X_train, y_train, cv= 10, scoring='accuracy')
# 26 ns ± 1.85 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)
# statement overall took 2.2s
model_dtc_cv_acc = cross_val_score(model_dtc, X_train, y_train, cv= 10, scoring='accuracy') 
# 23.7 ns ± 0.863 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)
# Statement overall took 5.9s 
# 
# Which classifier is the fastest and the slowest for you?
# 

# The KNN algorithm turns out to be the fastest and also the most accurate in terms of accuarcy. 
# Tree model was the second fastest model, with approximately 78% model accuarcy.
# Logistic regression performed the slowest, followed by SVC. 

###########  HW  ################
#%%
# First read in the datasets. 
import os
import numpy as np
import pandas as pd

# Import data
from sklearn.datasets import load_digits
digits = load_digits()
print("\nReady to continue.")

# check contents of digits: 
dir(digits)

#%%
from sklearn.model_selection import train_test_split


print("\nReady to continue.")

#%%
# What do they look like?
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
import matplotlib.pyplot as plt 
plt.gray() 
plt.matshow(digits.images[0])
plt.show() 
# The first row in the digits data set
plt.gray() 
plt.matshow(digits.images[-1])
plt.show() 
# The last row in the digits data set
print("\nReady to continue.")


#%% 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

print("\nReady to continue.")

# Using the df loaded from the the csv. 

#%%

# Sample code for cross validation
# We could use just the entire dataset (60k rows) for train-test split (90% training), 
# so that's 54000 data points for training. 
# Or we can start over, and get a set of X and y for CV here. 
# If we need to change size at some point, it's easier to do it here.
# NOTE: You might want to temporarily disable auto sleep/hibernation of your computer.
nmax = 8000 # nmax = 10000 # or other smaller values if your system resource is limited.
cvdigits = df.iloc[0:nmax,:]
X_cv = cvdigits.iloc[:,1:785] # 28x28 pixels = 784, so index run from 1 to 784 # remember that pandas iloc function like regular python slicing, do not include end number
print("cvdigits shape: ",cvdigits.shape)
print("X_cv shape: ",X_cv.shape)
y_cv = cvdigits.iloc[:,0]
#%%
# Logit Regression 
%timeit -r 1 print(f'\nLR CV accuracy score: { cross_val_score(model_logit, X_cv, y_cv, cv= 10, scoring="accuracy", n_jobs = -1) }\n')   

# For nmax = 2000:
# LR CV accuracy score: [0.86  0.88  0.865 0.835 0.83  0.82  0.85  0.85  0.905 0.875]
# 18 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)

# For nmax = 4000:
# LR CV accuracy score: [0.87   0.8675 0.835  0.8325 0.925  0.8675 0.87   0.885  0.9075 0.875 ]
# 2min 32s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)

# For nmax = 8000:
# LR CV accuracy score: [0.85875 0.84125 0.88    0.865   0.8775  0.8875  0.89125 0.89    0.8775   0.8625 ]
# Took more than 20mins

#%%
# the flag -r 1 is to tell timeit to repeat only 1 time to find the average time. The default is to repeat 7 times.
# I get something like below
# without n_jobs, quit: STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
# with n_jobs = -1, 
# nmax = 2000, it took ~ 6.81 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)
# LR CV accuracy score: [0.86  0.88  0.86  0.835 0.83  0.82  0.85  0.85  0.905 0.875]
#
# nmax = 4000, it took ~ 56 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)
# LR CV accuracy score: [0.86  0.88  0.86  0.835 0.83  0.82  0.85  0.85  0.905 0.875]
#
# nmax = 8000, it took ~ 6min 33s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)
# LR CV accuracy score: [0.86  0.88  0.86  0.835 0.83  0.82  0.85  0.85  0.905 0.875]
# 
# BUT if I hook up my laptop to external monitors as I usually do, even with 
# nmax = 2000, it took ~ 53.3 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)
# It is EIGHT times slower than before. The GPGPU is occupied with other tasks, and unable to 
# to dedicate on the task at hand.
# %%

#%%[markdown]
# You may use web search, notes, etc. 
# Do not use help from another human. If you use help from another student, 
# then I have no choice but to consider that student not a human, and will be 
# booted off my class immediately. You will also arrive at the same fate.
# 
#%%
import pandas as pd
import dm6103 as dm
df = dm.api_dsLand('Diet6wk','Person')
df.columns.values[3] = 'origweight'
df.info()

# The dataframe is on a person's weight 6 weeks after starting a diet. 
# Build these models:
# 
# 1. Using statsmodels library, build a linear model for the wight6weeks as a function of the other variables. Use gender and Diet as categorical variables. Print out the model summary. What is the r-squared value of the model?  
# 
#%%
import statsmodels.api as sm
from statsmodels.formula.api import glm
#%%

model_diet = glm(formula='weight6weeks ~ Age + Height + origweight + C(gender)+ C(Diet)', data=df, family=sm.families.Binomial())

model_diet = model_diet.fit()
print(model_diet.summary())
weight_weeks = model_diet.predict(df)
print(weight_weeks.head(10))

# The deviance of the model was 425700 (or negative two times Log-Likelihood-function)
print(-2*model_diet.llf)
# Compare to the null deviance
print(model_diet.null_deviance)
print(model_diet.resid_pearson)

#%%
# 2. Again using the statsmodels library, build a multinomial-logit regression model for the Diet (3 levels) as a function of the other variables. Use gender as categorical again. Print out the model summary. What is the  model's "psuedo r-squared" value?  
# 
# from statsmodels.formula.api import glm
#%%
from statsmodels.formula.api import mnlogit  # use this for multinomial logit in statsmodels library, instead of glm for binomial.
# Sample use/syntax:
# model = mnlogit(formula, data)

model_logit = mnlogit(formula = 'Diet ~ Age + Height + origweight + C(gender) + weight6weeks', data=df)

model_logit = model_logit.fit()
print(model_logit.summary())
diet_pred = model_logit.predict(df)
diet_pred.head(10)
# Model had a  Pseudo R-squared value of: 0.09026
#%%
# 3a. Use SKLearn from here onwards. 
# Use a 2:1 split, set up the training and test sets for the dataset, with Diet as y, and the rest as Xs. Use the seed value/random state as 1234 for the split.
#
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

df.head(10)
 
# Splitting the dataset into X (regressors) and y (target) sets 

X = df.drop('Diet', axis = 1)
y = df['Diet']

#%%
# Now, splitting into train/test sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6666, random_state=1234)


# Now generating the model:
model_sk = LogisticRegression(max_iter = 1000)
# Fitting the model.
model_sk.fit(X_train, y_train)
y_pred = model_sk.predict(X_test)
cm_model_sk = confusion_matrix(y_test, y_pred)
claf_report = classification_report(y_test, y_pred)

print(cm_model_sk)
print(claf_report)

# Accuracy of the model is 37% 
#%%
# 
# 3b. Build the corresponding logit regression as in Q2 here using sklearn. Train and score it. What is the score of your model with the training set and with the test set?
# 
import seaborn as sn
import matplotlib.pyplot as plt

model_sk2 = LogisticRegression(max_iter = 1000)
model_sk2.fit(X_train, y_train)
y_pred2 = model_sk.predict(X_test)
cm_model_sk2 = confusion_matrix(y_test, y_pred)
claf_report2 = classification_report(y_test, y_pred)

print(cm_model_sk2)
print(claf_report2)

sn.heatmap(cm_model_sk2, annot=True, fmt="d")
plt.show()
# Accuracy of the model is 37% 
#%%
# 4. Using the same training dataset, now use a 3-NN model, score the model with the training and test datasets. 
# 
from sklearn.neighbors import KNeighborsClassifier

model_knn = KNeighborsClassifier(n_neighbors= 3)
model_knn.fit(X_train, y_train)
y_pred_knn = model_knn.predict(X_test)
cm_model_knn = confusion_matrix(y_test, y_pred_knn)
claf_report_knn  = classification_report(y_test, y_pred_knn)
print(cm_model_knn)
print(claf_report_knn)

sn.heatmap(cm_model_knn, annot=True, fmt="d")
plt.show()

# Model produces a 41% Accuracy
#%%

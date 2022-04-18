#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dm6103 as dm

# Part I
titanic = dm.api_dsLand('Titanic', 'id')

# Part II
nfl = dm.api_dsLand('nfl2008_fga')
nfl.dropna(inplace=True)

#%% [markdown]

# # Part I  
# Titanic dataset - statsmodels
# 
# | Variable | Definition | Key/Notes  |  
# | ---- | ---- | ---- |   
# | survival | Survived or not | 0 = No, 1 = Yes |  
# | pclass | Ticket class | 1 = 1st, 2 = 2nd, 3 = 3rd |  
# | sex | Gender / Sex |  |  
# | age | Age in years |  |  
# | sibsp | # of siblings / spouses on the Titanic |  |  
# | parch | # of parents / children on the Titanic |  |  
# | ticket | Ticket number (for superstitious ones) |  |  
# | fare | Passenger fare |  |  
# | embarked | Port of Embarkation | C: Cherbourg, Q: Queenstown, S: Southampton  |  
# 
#%%
# ## Question 1  
# With the Titanic dataset, perform some summary visualizations:  
# 
# ### a. Histogram on age. Maybe a stacked histogram on age with male-female as two series if possible

# Checking the data set:
titanic.info()
# Grouping gender into men and female as two series. 
grouped = titanic.groupby('sex')
men_titanic = grouped.get_group('male')
female_titanic = grouped.get_group('female')

ax = men_titanic['age'].hist( bins=50, range=(1,100), stacked=True, color = 'blue', label = 'Male')
ax = female_titanic['age'].hist(bins=50, range=(1,100), stacked=True, color = 'pink', label = 'Female')
plt.setp(ax.get_xticklabels(),  rotation=45)
plt.xlabel('Gender Age')
plt.ylabel('Count')
  
plt.title('Male-Female Stacked Histogram on Age\n\n',
          fontweight ="bold")
  
plt.show()
#%%
# ### b. proportion summary of male-female, survived-dead  

men_titanic.describe()
count = men_titanic['survived'].value_counts()
print(count)

# 0: Dead (468)
# 1: Survived (109)
# Total: 577 (survived + dead)

# 19% male passengers survived
# -----------------------------------------------------------------
female_titanic.describe()
count2 = female_titanic['survived'].value_counts()
print(count2)

# 0: Dead (233)
# 1: Survived (81)
# Total: 314 (survived + dead)

# 25% female passengers survived.
#%%
# ### c. pie chart for “Ticketclass”  

# Checking for null values in pclass column:
titanic['pclass'].isnull().sum()
# No null values are found. 

# Creating a pie chart:
colors = sns.color_palette('pastel')[0:3]
ticket_grouped = titanic.groupby('pclass').size()
print(ticket_grouped)
plt.pie(ticket_grouped, labels = ['1st Class', '2nd Class', '3rd Class'], colors = colors, autopct='%.0f%%')
plt.show()

#%%
# ### d. A single visualization chart that shows info of survival, age, pclass, and sex.  

# ----- Can we show info in subplots (2,2) ? ---------


#%%

# ## Question 2  
# Build a logistic regression model for survival using the statsmodels library. As we did before, include the features that you find plausible. Make sure categorical variables are use properly. If the coefficient(s) turns out insignificant, drop it and re-build.  
import statsmodels.api as sm
from statsmodels.formula.api import glm

modelSurvival = glm(formula='survived ~ pclass + sex', data=titanic, family=sm.families.Binomial())

modelSurvival = modelSurvival.fit()
print(modelSurvival.summary())
survived_class_gender = modelSurvival.predict(titanic)
print(survived_class_gender.head(10))

# The deviance of the model was 827.20 (or negative two times Log-Likelihood-function)
print(-2*modelSurvival.llf)
# Compare to the null deviance
print(modelSurvival.null_deviance)
# 1186.655 

# This is a decrease of 360 with 2 variables. 
# Also, the small p values (almost 0) suggest that the model is better than the null model.

#%%
# ## Question 3  
# Interpret your result. What are the factors and how do they affect the chance of survival (or the survival odds ratio)? What is the predicted probability of survival for a 30-year-old female with a second class ticket, no siblings, 3 parents/children on the trip? Use whatever variables that are relevant in your model.  

# gender and pclass have been chosen as regressors in the above model. Both the variables have very low p values. 

# Checking actual survival status of such passenger:
female_titanic[(female_titanic['age'] == 30) & (female_titanic['parch'] == 3)]
# There is no passenger with 3 parents/children in the working data set. 

# However, there is a passenger with 3 siblings and 0 parents/children. 
female_titanic[(female_titanic['age'] == 30) & (female_titanic['pclass'] == 2) ]

# Passenger of 'id 727' in the data set has Survived (1)


 # Now comparing the model calculation with the actual result:
print(survived_class_gender[727])
#
# The model predicted 0.7979 or nearly 80% survival for the passenger. This is good in any case, since we are use lower cut-off values. 

#%%

# ## Question 4  
# Try three different cut-off values at 0.3, 0.5, and 0.7. What are the a) Total accuracy of the model b) The precision of the model (average for 0 and 1), and c) the recall rate of the model (average for 0 and 1)
survived_class_gender = pd.Series(survived_class_gender)
# Confusion matrix
# Define cut-off value
cut_off = 0.3
# Compute class predictions
survived_class_gender = np.where(survived_class_gender > cut_off, 1, 0)
#
# Make a cross table
print(pd.crosstab(titanic['survived'], survived_class_gender,
rownames=['Actual'], colnames=['Predicted'],
margins = True))
# Predicted    0    1  All
# Actual                  
# 0          391  158  549
# 1           64  278  342
# All        455  436  891

# Accuracy = 0.7508  (TP + TN) / Total
# Precision = 0.336 TP / (TP + FP)
# Recall Rate = 0.812 TP / (TP + FN)
#
#                         predicted 
#                   0                  1
# Actual 0   True Negative  TN      False Positive FP
# Actual 1   False Negative FN      True Positive  TP

cut_off = 0.5
# Compute class predictions
survived_class_gender = np.where(survived_class_gender > cut_off, 1, 0)
#
# Make a cross table
print(pd.crosstab(titanic['survived'], survived_class_gender,
rownames=['Actual'], colnames=['Predicted'],
margins = True))
# Predicted    0    1  All
# Actual                  
# 0          391  158  549
# 1           64  278  342
# All        455  436  891

cut_off = 0.7
# Compute class predictions
survived_class_gender = np.where(survived_class_gender > cut_off, 1, 0)
#
# Make a cross table
print(pd.crosstab(titanic['survived'], survived_class_gender,
rownames=['Actual'], colnames=['Predicted'],
margins = True))
# Predicted    0    1  All
# Actual                  
# 0          391  158  549
# 1           64  278  342
# All        455  436  891

# Same Result for all three cut-off values.

#%%[markdown]
# # Part II  
# NFL field goal dataset - SciKitLearn
# 
# | Variable | Definition | Key/Notes  |  
# | ---- | ---- | ---- |   
# | AwayTeam | Name of visiting team | |  
# | HomeTeam | Name of home team | |  
# | qtr | quarter | 1, 2, 3, 4 |  
# | min | Time: minutes in the game |  |  
# | sec | Time: seconds in the game |  |  
# | kickteam | Name of kicking team |  |  
# | distance | Distance of the kick, from goal post (yards) |  |  
# | timerem | Time remaining in game (seconds) |  |  
# | GOOD | Whether the kick is good or no good | If not GOOD: |  
# | Missed | If the kick misses the mark | either Missed |  
# | Blocked | If the kick is blocked by the defense | or blocked |  
# 
#%% 
# ## Question 5  
# With the nfl dataset, perform some summary visualizations.  
# 

# Creating a subset of teams that are home teams and won the game. 
home_win = nfl[nfl['AwayTeam'] == nfl['def']]

homewin_counts = (home_win['HomeTeam'].value_counts())
homewin_counts.plot(kind = 'bar', color = 'black')
# We can see that NYG has won the most home games MIA has won the least home games in the year 2008 compared to all teams. 

# Creating a subset of teams in which away teams and won the game. 
home_loss = nfl[nfl['HomeTeam'] == nfl['def']]

homeloss_counts = (home_loss['HomeTeam'].value_counts())
homeloss_counts.plot(kind = 'bar', color = 'lightred')

# We can see that the team that lost the most number of games at home was DAL and the team that lost the least numbers of games at home was BAL. 

#%%
# ## Question 6  
# Using the SciKitLearn library, build a logistic regression model overall (not individual team or kicker) to predict the chances of a successful field goal. What variables do you have in your model? 
# 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#%%
# Separating the target variable(GOOD) and the regressors into two different data sets. 

X = nfl.drop('GOOD', axis = 'columns') # regressor set
y = nfl['GOOD'] # target set

#%%
# Creating test/train data sets from the above set using sklearn train_test_split: 80% train, 20% test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 15)

print(y_train.value_counts()) # 80% as training set   
print(y_test.value_counts()) # 20% as test set

#%%

# creating a logistic model:

nflmodel = LogisticRegression(max_iter= 100)

nflmodel.fit(X_train, y_train) 
#%%a

# 
# ## Question 7  
# Someone has a feeling that home teams are more relaxed and have a friendly crowd, they should kick better field goals. Use your model to find out if that is subtantiated or not. 
# 
#  
# ## Question 8    
# From what you found, do home teams and road teams have different chances of making a successful field goal? If one does, is that true for all distances, or only with a certain range?
# 


# %%
# titanic.dropna()


# %%

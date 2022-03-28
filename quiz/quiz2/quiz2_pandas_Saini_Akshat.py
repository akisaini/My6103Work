#%%[markdown]
# Quiz 2
# Name:  Akshat Saini
# 
# You may use web search, notes, etc. 
# Do not use help from another human. If you use help from another student, 
# then I have no choice but to consider that student not a human, and will be 
# booted off my class immediately. You will also arrive at the same fate.
#
# From the titanic dataframe: 
# Complete the tasks below without importing any libraries except pandas (and dm6103).
# 
# 1. what is the total fare paid by all the passengers on board? 
# 
# 2. create a boolean array/dataframe for female passengers. Use broadcasting and filtering 
# to obtain a subset of females, and find average age of the female passengers on board 
# 
# 3. create a boolean array/dataframe for survived passengers. Use broadcasting and filtering 
# to obtain a subset of survivers, and find the average age of the survived passengers on board? 
# 
# 4. What is the average age of the female passengers who survived? 
# 
# survival : Survival,	0 = No, 1 = Yes
# pclass : Ticket class, 1 = 1st, 2 = 2nd, 3 = 3rd
# sex : Gender / Sex
# age : Age in years
# sibsp : # of siblings / spouses on the Titanic
# parch : # of parents / children on the Titanic
# ticket : Ticket number (for superstitious ones)
# fare : Passenger fare
# embarked : Port of Embarkment	C: Cherbourg, Q: Queenstown, S: Southampton

import pandas as pd
import dm6103 as dm
df = dm.api_dsLand('Titanic','id')
print(df.info())
print(df.head())


#%%
# 1. what is the total fare paid by all the passengers on board? 
fare_sum = df.loc[:,'fare'].sum()
print(fare_sum)
#Total sum paid by the passengers is 28693.94
#%%
# 2. create a boolean array/dataframe for female passengers. Use broadcasting and filtering 
# to obtain a subset of females, and find average age of the female passengers on board 
# 
fem_bool = df.loc[:,'sex'] == 'female'
fem = df[fem_bool]
fem['age'].mean()
#Average age of females was 23.204
#%%
# 3. create a boolean array/dataframe for survived passengers. Use broadcasting and filtering 
# to obtain a subset of survivers, and find the average age of the survived passengers on board? 
# 
surv_bool = df.loc[:,'survived'] == 1
survived = df[surv_bool]
survived['age'].mean()

#Average age of survived passengers was 24.03. This includes both females and males. 
#%%
# 4. What is the average age of the female passengers who survived? 
# 
fem_surv_bool = fem.loc[:,'survived'] == 1
fem[fem_surv_bool].loc[:,'age'].mean()
#Average age of suvived female passengers was 24.39 
# %%


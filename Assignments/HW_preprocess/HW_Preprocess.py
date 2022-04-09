# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dm6103 as dm

# The dataset is obtained from 
# https://gssdataexplorer.norc.org 
# for you here. But if you are interested, you can try get it yourself. 
# create an account
# create a project
# select these eight variables: 
# ballot, id, year, hrs1 (hours worked last week), marital, 
# childs, income, happy, 
# (use the search function to find them if needed.)
# add the variables to cart 
# extract data 
# name your extract
# add all the 8 variables to the extract
# Choose output option, select only years 2000 - 2018 
# file format Excel Workbook (data + metadata)
# create extract
# It will take some time to process. 
# When it is ready, click on the download button. 
# you will get a .tar file
# if your system cannot unzip it, google it. (Windows can use 7zip utility. Mac should have it (tar function) built-in.)
# Open in excel (or other comparable software), then save it as csv
# So now you have Happy table to work with
#
# When we import using pandas, we need to do pre-processing like what we did in class
# So clean up the columns. You can use some of the functions we defined in class, like the total family income, and number of children. 
# Other ones like worked hour last week, etc, you'll need a new function. 
# Happy: change it to numeric codes (ordinal variable)
# Ballot: just call it a, b, or c 
# Marital status, it's up to you whether you want to rename the values. 
# 
#
# After the preprocessing, make these plots
# Box plot for hours worked last week, for the different marital status. (So x is marital status, and y is hours worked.) 
# Violin plot for income vs happiness, 
# (To use the hue/split option, we need a variable with 2 values/binomial, which 
# we do not have here. So no need to worry about using hue/split for this violinplot.)
# Use happiness as numeric, make scatterplot with jittering in both x and y between happiness and number of children. Choose what variable you want for hue/color.
# If you have somewhat of a belief that happiness is caused/determined/affected by number of children, or the other 
# way around (having babies/children are caused/determined/affected by happiness), then put the dependent 
# variable in y, and briefly explain your choice.

dfhappy = dm.api_dsLand('Happy') 


#%%
# Checking the different categories in each variable:
print(dfhappy.groupby(['marital']).count().reset_index()) 
# 6 separate categories:
# Divorced: presented as '1'
# Married: presented as '2'
# Never-Married: presented as '3'
# Separated: presented as '4'
# Widowed: presented as '5'
# No answer: presented as 'NA'
# %%
print(dfhappy.groupby(['income']).count().reset_index()) 
# 15 different categories:
# No answer: presented as '0'
# $8000 to 9999: presented as '1'
# $15000 - 19999: presented as '2'
# $25000 or more: presented as '3'
# $20000 - 24999: presented as '4'
# $10000 - 14999: presented as '5'
# Refused: presented as '6'
# $1000 to 2999: presented as '7' 
# Don't know: presented as '8'
# Lt $1000 or (less than $1000): presented as '9'
# $5000 to 5999: presented as '10'
# $7000 to 7999: presented as '11'
# $3000 to 3999: presented as '12'
# $4000 to 4999: presented as '13'
# $6000 to 6999: presented as '14'

# %%
print(dfhappy.groupby(['childs']).count().reset_index()) 
# Total 10 categories:
# 0: presented as '1'
# 1: presented as '2'
# 2: presented as '3'
# 3: presented as '4'
# 4: presented as '5'
# 5: presented as '6'
# 6: presented as '7'
# 7: presented as '8'
# Eight or m: presented as '9'
# Dk na: presented as 'NA'
# %%
print(dfhappy.groupby(['happy']).count().reset_index()) 
# 6 differnet categories:
# Dont know - presented as '1'
# Not applicable - presented as '2'
# Not too happy - presented as '3'
# Pretty happy - presented as '4'
# Very Happy - presented as '5'
# No answer - presented as 'NA'
# %%
print(dfhappy.groupby(['ballet']).count().reset_index()) 
# 4 differnet categories:
# Ballot a - presented as 'a'
# Ballot b - presented as 'b'
# Ballot c - presented as 'c'
# Ballot d - presented as 'd'


#%%
# Changing column data to ordinal/categorical values matching different categories:
dfhappy['ballet'].replace(to_replace=['Ballot a', 'Ballot b', 'Ballot c', 'Ballot d'], value=['a', 'b', 'c', 'd'], inplace=True)

dfhappy['happy'].replace(to_replace=['Don\'t know', 'Not applicable', 'Not too happy', 'Pretty happy', 'Very happy', 'No answer'], value=['1', '2', '3', '4', '5', pd.NA], inplace=True)
                          
dfhappy['childs'].replace(to_replace=['0','1','2','3','4','5','6','7','Eight or m','Dk na'], value=['1', '2', '3', '4', '5', '6', '7', '8', '9', pd.NA], inplace=True)


dfhappy['marital'].replace(to_replace=['Divorced', 'Married', 'Never married', 'Separated', 'Widowed', 'No answer'], value=['1', '2', '3', '4', '5', pd.NA], inplace=True)                    
                                                  
#%%
# Converting column datatype to type categoric and hrs1 and income data type to numeric:
dfhappy['marital'] = dfhappy['marital'].astype('category')
dfhappy['childs'] = dfhappy['childs'].astype('category')
dfhappy['happy'] = dfhappy['happy'].astype('category')
dfhappy['ballet'] = dfhappy['ballet'].astype('category')
dfhappy['hrs1'] = pd.factorize(dfhappy['hrs1'])[0]
dfhappy['income'] = pd.factorize(dfhappy['income'])[0]
# Verify data type pf variables:
dfhappy.info()

# %%
# Checking the updated data set now: (NA values are still included.)
dfhappy
# %%
# Plots using matplotlib: 
# Box plot between marital status and hrs worked:
sns.set_style("whitegrid")
sns.boxplot(x= 'marital', y = 'hrs1', data = dfhappy)
plt.show()
# %%
# Violin plot for income vs happiness: 
sns.set_style("whitegrid")
sns.violinplot(x = 'income', y = 'happy', data = dfhappy)
plt.show()

# %%

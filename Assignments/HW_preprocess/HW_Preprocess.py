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
# No answer
# Divorced
# Married
# Never-Married
# Separated
# Widowed

# %%
print(dfhappy.groupby(['income']).count().reset_index()) 
# 15 different categories:
# No answer
# $1000 to 2999
# $10000 - 14999
# $15000 - 19999
# $20000 - 24999
# $25000 or more
# $3000 to 3999
# $4000 to 4999
# $5000 to 5999
# $6000 to 6999
# $7000 to 7999
# $8000 to 9999
# Don't know
# Lt $1000 or (less than $1000)
# Refused

# %%
print(dfhappy.groupby(['childs']).count().reset_index()) 
# Total 10 categories:
# 0: presented as '0'
# 1: presented as '1'
# 2: presented as '2'
# 3: presented as '3'
# 4: presented as '4'
# 5: presented as '5'
# 6: presented as '6'
# 7: presented as '7'
# Eight or m: presented as '8'
# Dk na: presented as pd.NA
# %%
print(dfhappy.groupby(['happy']).count().reset_index()) 
# 6 differnet categories:
# No answer - presented as '0'
# Dont know - presented as '1'
# Not applicable - presented as '2'
# Not too happy - presented as '3'
# Pretty happy - presented as '4'
# Very Happy - presented as '5'

# %%
print(dfhappy.groupby(['ballet']).count().reset_index()) 
# 4 differnet categories:
# Ballot a - presented as 'a'
# Ballot b - presented as 'b'
# Ballot c - presented as 'c'
# Ballot d - presented as 'd'


#%%
# Changing column data to ordinal/numeric values matching different categories:
dfhappy['ballet'].replace(to_replace=['Ballot a', 'Ballot b', 'Ballot c', 'Ballot d'], value=['a', 'b', 'c', 'd'], inplace=True)

dfhappy['happy'].replace(to_replace=['Don\'t know', 'Not applicable', 'Not too happy', 'Pretty happy', 'Very happy', 'No answer'], value=['1', '2', '3', '4', '5', '0'], inplace=True)
dfhappy['happy'] = pd.to_numeric(dfhappy['happy'])
                          

#dfhappy['income'].replace(to_replace=['$1000 to 2999', '$10000 - 14999', '$15000 - 19999', '$20000 - 24999', '$25000 or more', '$3000 to 3999', '$4000 to 4999', '$5000 to 5999', '$6000 to 6999', '$7000 to 7999', '$8000 to 9999', 'Don\'t know', 'Lt $1000', 'Refused', 'No answer'], value=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '0'], inplace=True)


# Fixing the 'childs' column by converting the 'Eight or more' value to 8 and 'Dk na' value to NA:

dfhappy['childs'].replace(to_replace=['Dk na', 'Eight or m'], value=[pd.NA, '8'], inplace=True)
dfhappy['childs'] = pd.to_numeric(dfhappy['childs'])


# Fixing the 'hrs1' column by converting the 'Dont Know' value to  and 'Not applicable' value to  and 'No answer' value to NA:

dfhappy['hrs1'].replace(to_replace=['Don\'t know', 'Not applicable', 'No answer'], value=[pd.NA, pd.NA, pd.NA], inplace=True)
dfhappy['hrs1'] = pd.to_numeric(dfhappy['hrs1'])


#%%
# Verify data type of variables:
dfhappy.info()
dfhappy
# %%
#Plotting:

# Plots using seaborn: 
# Box plot between marital status and hrs worked:
sns.set_style("whitegrid")
sns.boxplot(x= 'marital', y = 'hrs1', data = dfhappy)
plt.show()
# %%
# Happiness categories:
# Dont know - presented as '1'
# Not applicable - presented as '2'
# Not too happy - presented as '3'
# Pretty happy - presented as '4'
# Very Happy - presented as '5'
# No answer - presented as 'NA'
#-----------------------------------------------------------
# Violin plot for income vs happiness:
# Is income dependent on happiness? x/independent variable = happiness, y/dependent variable = income
sns.set_style("whitegrid")
sns.violinplot(x = 'happy', y = 'income', data = dfhappy, split =True)
plt.show()

# %%
# Children categories:
# 0: presented as '0'
# 1: presented as '1'
# 2: presented as '2'
# 3: presented as '3'
# 4: presented as '4'
# 5: presented as '5'
# 6: presented as '6'
# 7: presented as '7'
# Eight or m: presented as '8'
# Dk na: presented as pd.NA
#----------------------------------------------------------
# Happiness vs Number of Children - jitter plot with marital hue

plot = sns.stripplot(x='happy', y='childs', hue='marital', data=dfhappy, palette='ocean', jitter=True, edgecolor='none', alpha=.60 )
plot.get_legend().set_visible(False)
# Put legend outside the fig 
plt.legend(bbox_to_anchor=(1,0), loc="lower left", labelcolor='black', facecolor='white', edgecolor='black', fontsize='large')
plt.title('Happiness vs Number of Children | Marital Status as Hue')
# show plot
plt.show()
sns.despine()

# %%

# happy vs childs(dependent)
# Checking if number of children is dependent on level of happiness.

sns.boxplot(x ='happy', y ='childs', data = dfhappy, palette ='plasma', hue = 'marital')
plt.legend(bbox_to_anchor=(1,0), loc="lower left", labelcolor='black', facecolor='white', edgecolor='black', fontsize='large')
# show plot
plt.show()
sns.despine()

# Being happy doesnt necessarily lead to more/less children. Below observations have been made from the above plot:

# Married people who are 'very happy'/5 have a mean of 2 children. 
# People who never married and who are 'very happy'/5 usually have between 0 or 1 children.
# People who are divorced and are 'very happy'/5 have in-between 1 and 3 children with mean of 2. 
# People who are 'not too happy'/3 and who have never married have between 0 and 2 children. 
# People who are 'not too happy'/3 and who are divorced have between 1 and 3 children with mean of 2. This is similar to people who are 'very happy' and 'divorced'.  
# People who are 'not too happy'/3 and who are married have between 1 and 3 children with mean of 2. This is again similar to 'very happy'/5 people who are married. 

#summary: People who are 'not too happy' had children in the same range as compared to people who are 'very happy'. 
# %%

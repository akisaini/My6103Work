# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dm6103 as dm

world1 = dm.api_dsLand('World1', 'id')
world2 = dm.api_dsLand('World2', 'id')

print("\nReady to continue.")


#%% [markdown]
# # Two Worlds 
# 
# I was searching for utopia, and came to this conclusion: If you want to do it right, do it yourself. 
# So I created two worlds. 
#
# Data dictionary:
# * age00: the age at the time of creation. This is only the population from age 30-60.  
# * education: years of education they have had. Education assumed to have stopped. A static data column.  
# * marital: 0-never married, 1-married, 2-divorced, 3-widowed  
# * gender: 0-female, 1-male (for simplicity)  
# * ethnic: 0, 1, 2 (just made up)  
# * income00: annual income at the time of creation   
# * industry: (ordered with increasing average annual salary, according to govt data.)   
#   0. leisure n hospitality  
#   1. retail   
#   2. Education   
#   3. Health   
#   4. construction   
#   5. manufacturing   
#   6. professional n business   
#   7. finance   
# 
# 
# Please do whatever analysis you need, convince your audience both, one, or none of these 
# worlds is fair, or close to a utopia. 
# Use plots, maybe pivot tables, and statistical tests (optional), whatever you deem appropriate 
# and convincing, to draw your conclusions. 
# 
# There are no must-dos (except plots), should-dos, cannot-dos. The more convenicing your analysis, 
# the higher the grade. It's an art.
#

#%% [markdown]
# *Basic EDA:*
#
# Using the info function to find the general datatypes and information of all the variables in world1 and world2 data sets:
#
world1.info()
world2.info()
#
# using the describe method to compare 'mean' values in differenr variables:
#
world1.describe()
world2.describe()
# The mean age in both the worlds is 44 years for the population. 
# Years of education is 15 years in both worlds as well. 
#
#
# grouping gender and comparing the variables in different worlds using aggregate function 'sum' and 'mean':
grouped1 = world1.groupby('gender').agg([np.sum, np.mean])
print(grouped1)
#
# In world1 we can see that average age for both men and women is 
# 44 years and both the genders are educated an average of 15 years. 
# It can be noted however that men earn an average of 11000 more than women annually in world1. *So there is gender pay gap.* 
#
#
# Now grouping gender and comparing the variables in world2:
grouped2 = world2.groupby('gender').agg([np.sum, np.mean])
print(grouped2)
#
# In wordl2 we can note that the average age of both men and women is 44 years and both the genders go through an overall education of 15 years(avg) as calculated. 
# There is relatively no pay gap between the two genders compared to world1. And both men and women tend to earn an average amount of 60,000 anually. 
#
#
#
# Creating subsets based on gender and then comparing different variables between the two worlds. 
#
# world1: creating gender subsets
gen_bool = world1.loc[:,'gender'] == 1
world1_men = world1.loc[:,:][gen_bool]
gen_bool2 = world1.loc[:,'gender'] == 0
world1_women = world1.loc[:,:][gen_bool2]
# In world1 there are *11857 men* and *12143 women*. 
#
#
# world2: creating gender subsets
gen1_bool = world2.loc[:,'gender'] == 1
world2_men = world2.loc[:,:][gen1_bool]
gen2_bool2 = world2.loc[:,'gender'] == 0
world2_women = world2.loc[:,:][gen2_bool2]
# In world2 there are *11690 men* and *12310 women*.
#
# Population is distributed nearly equally. 

# %%
# Now creating gender wise plots to understand the situation at a
# deeper level. 
# %%
plt.scatter(world1_men['education'], world1_men['income00'], label = 'world 1')
plt.scatter(world2_men['education'], world2_men['income00'], label = 'world 2')
plt.legend()
plt.title('Income vs Education(years) in both worlds for men')
plt.show()

# Men in world 1 can get paid more with the same level of education as men in world 2. 

# %%
plt.scatter(world1_women['education'], world1_women['income00'], label = 'world 1')
plt.scatter(world2_women['education'], world2_women['income00'], label = 'world 2')
plt.legend()
plt.title('Income vs Education(years) in both worlds for women')
plt.show()
 # Women in world 2 are paid more with the same level of eduction usually. 

# %%
# Comparing marital status:
fig = plt.figure()
ax1= fig.add_subplot(1, 2, 1)
ax1.hist(world1['marital'], color ='green', label = 'World 1')
ax1.legend()
ax2 = fig.add_subplot(1,2,2)
ax2.hist(world2['marital'], color ='orange', label = 'World 2')
ax2.legend()
fig.tight_layout()
fig.suptitle("Histogram of marital status in the two worlds")
plt.subplots_adjust(top=0.85)
plt.show()

# Marital status compared between the two worlds is extremely identical. In world 1 there are some men who are still unmarried(0) or divorced(2) compared to world 2. 


# Comparing industry division between the two worlds:
x = np.arange(0, 8, 1)
plt.subplot(1,2,1)
plt.hist(world1['industry'], color ='red', label = 'World 1', bins = 15, edgecolor = 'blue')
plt.xticks(x)
plt.legend()
plt.title('Histogram of industry division in the two worlds', loc = 'center')
plt.subplot(1,2,2)
plt.hist(world2['industry'], color ='black', label = 'World 2', bins = 15, edgecolor = 'blue')
plt.xticks(x)
plt.legend()
plt.subplots_adjust(top=.80, wspace = 1.00)
plt.show()

# plot to compare gender prefrences/role in employment in world1:
x = np.arange(0, 8, 1)
plt.subplot(1,2,1)
plt.hist(world1_men['industry'], color ='orange', label = 'Men', bins = 15, edgecolor = 'blue')
plt.xticks(x)
plt.legend()
plt.title('Histogram of gender division in employment in world 1', loc = 'center')
plt.subplot(1,2,2)
plt.hist(world1_women['industry'], color ='black', label = 'Women', bins = 15, edgecolor = 'blue')
plt.xticks(x)
plt.legend()
plt.subplots_adjust(top=.80, wspace = 1.00)
plt.show()

# The plots show similar distribution between industry employment when comparing the two worlds. Retail(1), health(3) and professional n business(6) tend to be the biggest employers in both the worlds. 

# --------------------------------------------------------------

# Aside from this, in world1 it can be noted that 'Construction'(4) is very uncommon among women and mostly a male dominated sector. 
# Retail(1), Education(2) and Health(3) are very popular among women, being one of the bigger employers, as compared to men. 
# Professional n business(6), Construction(4) and finance(7) tend to be dominated by males in world 1. 


x = np.arange(0, 8, 1)
plt.subplot(1,2,1)
plt.hist(world2_men['industry'], color ='gold', label = 'Men', bins = 15, edgecolor = 'blue')
plt.xticks(x)
plt.legend()
plt.title('Histogram of gender division in employment in world 2', loc = 'center')
plt.subplot(1,2,2)
plt.hist(world2_women['industry'], color ='saddlebrown', label = 'Women', bins = 15, edgecolor = 'blue')
plt.xticks(x)
plt.legend()
plt.subplots_adjust(top=.80, wspace = 1.00)
plt.show()

# In world 2, it can be noted that:

# Employment between men and women is more equally distributed as compared to world 1. 
# Even in businesses like contruction(4), women are equally employed in world 2. 
# Retail(1), Education(2) and Health(3) are very good souces of employment for men in world 2 contrary to world 1. 
# Leisure n hospitality(0) is a little more common among women compared to men in world 2.
# Comparing the two worlds based on employment sectors, world 2 is better distributed between the genders than world 1 and is more welcoming for everyone.  


# %%

#Summary:

# Population and other variables like marital, ethinicity, age and education are very identical between the two worlds. 
#
# Gender income gap exists in world 1.
# Industry gap between genders can be noticed in world 1. 
# Avg income is same for men and women in world 2. 
# Income gap of 11k in world 1. 
# Otherwise average income is identical between the two worlds.(60k)
# More infomation might have been useful such as crime statistics, weather, and happiness quotient/life expectency etc. But solely based on the given information, World 2 is probably close to what we are seeking. But World 1 is still very livable. 
# We create our own utopia with family and friends and that can be created anywhere, even in a jungle. 
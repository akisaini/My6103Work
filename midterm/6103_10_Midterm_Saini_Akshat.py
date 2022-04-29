# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
from statistics import mean
from sys import platlibdir
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
#----------------------------------------------------------------------
# *What is a utopian society?*
#
# The concept of utopia refers to a perfect or ideal civilization. It typically describes an imaginary community or society that possesses highly desirable or nearly perfect qualities for its members. 
#
# There is no consensus as to exactly what a utopia should look like, but values like freedom, security, equality, and enlightened thinking are all common features in utopian discussions and literature.
#
# *Utopian Society Ideals:*
# Theoretical utopian societies usually provide their citizens with equal rights, a life without fear, economic security, and collective or government-provided welfare for all.
#
# Some common utopia examples to relate with include scenarios like:  
# * The Federation in the Star Trek series. 
# * The Capital in The Hunger Games series, a place of luxury and freedom.
# * England in Aldous Huxley's Brave New World, a place with no wars or hunger but also no emotion.
#
# In the '2021 Best Countires in the World' Ranking conducted by USNews, the overall ranking of best countries measured global performace on a variety of metrics, amongst which 'literacy' was one of the top criterias, followed by 'equality/racial equality' and an 'economically stable job market'.
#
# In our model, we will try to focus more on the above three aspects to compare the 2 different worlds to the definition of Utopia. 
#
#
#%%
# Basic EDA:
#
# Using the info function to find the general datatypes and information of all the variables in world1 and world2 data sets:
#world1.info()
#world2.info()
#
# using the describe method to compare 'mean' values in differenr variables:
#world1.describe()
#world2.describe()
#
# The mean age in both the worlds is 44 years for the population. 
# Years of education is 15 years in both worlds as well. 
#
# We could use pivot table here to capture gender mean values for certain variables like age00 or income00 etc. , but grouping should work better. 
gender_pivot = pd.pivot_table(data = world1, index = 'gender', values = 'age00', aggfunc= 'mean')
#print(gender_pivot)
# grouping gender and comparing the variables in different worlds using aggregate function 'sum' and 'mean':
grouped1 = world1.groupby('gender').agg([np.sum, np.mean])
#print(grouped1)
#
# In world1 we can see that average age for both men and women is 
# 44 years and both the genders are educated an average of 15 years. 
# It can be noted however that men earn an average of 11000 more than women annually in world1. *So there is gender pay gap.* 
#
# Now grouping gender and comparing the variables in world2:
grouped2 = world2.groupby('gender').agg([np.sum, np.mean])
#print(grouped2)
#
# In wordl2 we can note that the average age of both men and women is 44 years and both the genders go through an overall education of 15 years(avg) as calculated. 
# There is relatively no pay gap between the two genders compared to world1. And both men and women tend to earn an average amount of 60,000 anually. 
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
#%%
# Define ethnic data sets:
# Creating a dataset that is grouped in terms of column ethnic: 

ethnic_grouped1 = world1.groupby('ethnic')
ethnic_grouped2 = world2.groupby('ethnic')
w1_ethnic_0 = ethnic_grouped1.get_group(0)
w1_ethnic_1 = ethnic_grouped1.get_group(1)
w1_ethnic_2 = ethnic_grouped1.get_group(2)
w2_ethnic_0 = ethnic_grouped2.get_group(0)
w2_ethnic_1 = ethnic_grouped2.get_group(1)
w2_ethnic_2 = ethnic_grouped2.get_group(2)

#define data
data = [w1_ethnic_0.shape[0], w1_ethnic_1.shape[0], w1_ethnic_2.shape[0]]
labels = ['ethnic_0', 'ethnic_1', 'ethnic_2']

#define Seaborn color palette to use
colors = sns.color_palette('pastel')[0:5]

#create pie chart
plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%')
plt.title('Pie chart comparing population of different ethnicities in World 1')
plt.show()

data = [w2_ethnic_0.shape[0], w2_ethnic_1.shape[0], w2_ethnic_2.shape[0]]
labels = ['ethnic_0', 'ethnic_1', 'ethnic_2']

#define Seaborn color palette to use
colors = sns.color_palette('husl')[3:6]

#create pie chart
plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%')
plt.title('Pie chart comparing population of different ethnicities in World 2')
plt.show()

# Different ethnicities are equally divided in terms of population in both the worlds! No one ethnicity is dominant in either world. 

#%%
# Now, comparing the average statistics - like age, education years, industry preference different ethnicities - including both genders. 

# 1) Age statistics: 

data = {'Ethnic_0':w1_ethnic_0['age00'].mean(), 'Ethnic_1':w1_ethnic_1['age00'].mean(), 'Ethnic_2':w1_ethnic_2['age00'].mean()}
courses = list(data.keys())
values = list(data.values())
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(courses, values, color ='green',
        width = 0.3)
plt.title('Bar Plot comparing avg. Age of different ethnicities in World 1', fontsize = 18)
plt.show()


data = {'Ethnic_0':w2_ethnic_0['age00'].mean(), 'Ethnic_1':w2_ethnic_1['age00'].mean(), 'Ethnic_2':w2_ethnic_2['age00'].mean()}
courses = list(data.keys())
values = list(data.values())
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(courses, values, color ='teal',
        width = 0.3)
plt.title('Bar Plot comparing avg. Age of different ethnicities in World 2',fontsize = 18)
plt.show()
# Average Age between ethnicities is same for all three ethnicities in both the worlds!

#%%
# 2) Education years statistics: 

data = {'Ethnic_0':w1_ethnic_0['education'].mean(), 'Ethnic_1':w1_ethnic_1['education'].mean(), 'Ethnic_2':w1_ethnic_2['education'].mean()}
courses = list(data.keys())
values = list(data.values())
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(courses, values, color ='maroon',
        width = 0.3)
plt.grid(b = True, color ='grey',
        linestyle ='-.', linewidth = 0.5,
        alpha = 0.2)
plt.title('Bar Plot comparing avg. education of different ethnicities in World 1')
plt.show()


data = {'Ethnic_0':w2_ethnic_0['education'].mean(), 'Ethnic_1':w2_ethnic_1['education'].mean(), 'Ethnic_2':w2_ethnic_2['education'].mean()}
courses = list(data.keys())
values = list(data.values())
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(courses, values, color ='orange',
        width = 0.3)
plt.grid(b = True, color ='grey',
        linestyle ='-.', linewidth = 0.5,
        alpha = 0.2)
plt.title('Bar Plot comparing avg. education of different ethnicities in World 2')
plt.show()

# Average Education between ethnicities is same for all three ethnicities in both the worlds!
#%%
# Lets Check industry preference within ethnicities. This is to check if one ethnicity prefers a particular industry over another when compared to a different ethnicity in world 1. 

#   0. leisure n hospitality  
#   1. retail   
#   2. Education   
#   3. Health   
#   4. construction   
#   5. manufacturing   
#   6. professional n business   
#   7. finance   

data = [w1_ethnic_0[w1_ethnic_0['industry']==0].shape[0], w1_ethnic_0[w1_ethnic_0['industry']==1].shape[0], w1_ethnic_0[w1_ethnic_0['industry']==2].shape[0], w1_ethnic_0[w1_ethnic_0['industry']==3].shape[0], w1_ethnic_0[w1_ethnic_0['industry']==4].shape[0], w1_ethnic_0[w1_ethnic_0['industry']==5].shape[0], w1_ethnic_0[w1_ethnic_0['industry']==6].shape[0], w1_ethnic_0[w1_ethnic_0['industry']==7].shape[0]]
labels = ['Leisure n Hospitality', 'Retail', 'Education' ,'Health', 'Construction', 'Manufacturing', 'Professional n Business', 'Finance']

#define Seaborn color palette to use
colors = sns.color_palette('husl')[0:7]

#create pie chart
plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%')
plt.title('Pie chart showing Ethnicity 0\'s preference in Industry in World 1')
plt.show()

data = [w1_ethnic_1[w1_ethnic_1['industry']==0].shape[0], w1_ethnic_1[w1_ethnic_1['industry']==1].shape[0], w1_ethnic_1[w1_ethnic_1['industry']==2].shape[0], w1_ethnic_1[w1_ethnic_1['industry']==3].shape[0], w1_ethnic_1[w1_ethnic_1['industry']==4].shape[0], w1_ethnic_1[w1_ethnic_1['industry']==5].shape[0], w1_ethnic_1[w1_ethnic_1['industry']==6].shape[0], w1_ethnic_1[w1_ethnic_1['industry']==7].shape[0]]
labels = ['Leisure n Hospitality', 'Retail', 'Education' ,'Health', 'Construction', 'Manufacturing', 'Professional n Business', 'Finance']

#define Seaborn color palette to use
colors = sns.color_palette('rocket')[0:7]

#create pie chart
plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%')
plt.title('Pie chart showing Ethnicity 1\'s preference in Industry in World 1')
plt.show()

data = [w1_ethnic_2[w1_ethnic_2['industry']==0].shape[0], w1_ethnic_2[w1_ethnic_2['industry']==1].shape[0], w1_ethnic_2[w1_ethnic_2['industry']==2].shape[0], w1_ethnic_2[w1_ethnic_2['industry']==3].shape[0], w1_ethnic_2[w1_ethnic_2['industry']==4].shape[0], w1_ethnic_2[w1_ethnic_2['industry']==5].shape[0], w1_ethnic_2[w1_ethnic_2['industry']==6].shape[0], w1_ethnic_2[w1_ethnic_2['industry']==7].shape[0]]
labels = ['Leisure n Hospitality', 'Retail', 'Education' ,'Health', 'Construction', 'Manufacturing', 'Professional n Business', 'Finance']

#define Seaborn color palette to use
colors = sns.color_palette('viridis')[0:7]

#create pie chart
plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%')
plt.title('Pie chart showing Ethnicity 2\'s preference in Industry in World 1')
plt.show()

# It can be seen that ethnicity 0 has more people working in Health, Retail and Education sectors than manufacturing or Finance. 

# Ethnicity 1 has more people working in Hospitality, Health, Retail and Construction sectors. And very little in Manufacturing or Finance. 

# Ethnicity 2 has majority people working in 'Professional n Business', Manufacturing and Finance. And just 1% people working in Construction.   

#%% 

# Above analysis in world 2:

data = [w2_ethnic_0[w2_ethnic_0['industry']==0].shape[0], w2_ethnic_0[w2_ethnic_0['industry']==1].shape[0], w2_ethnic_0[w2_ethnic_0['industry']==2].shape[0], w2_ethnic_0[w2_ethnic_0['industry']==3].shape[0], w2_ethnic_0[w2_ethnic_0['industry']==4].shape[0], w2_ethnic_0[w2_ethnic_0['industry']==5].shape[0], w2_ethnic_0[w2_ethnic_0['industry']==6].shape[0], w2_ethnic_0[w2_ethnic_0['industry']==7].shape[0]]
labels = ['Leisure n Hospitality', 'Retail', 'Education' ,'Health', 'Construction', 'Manufacturing', 'Professional n Business', 'Finance']

#define Seaborn color palette to use
colors = sns.color_palette('husl')[0:7]

#create pie chart
plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%')
plt.title('Pie chart showing Ethnicity 0\'s preference in Industry in World 2')
plt.show()

data = [w2_ethnic_1[w2_ethnic_1['industry']==0].shape[0], w2_ethnic_1[w2_ethnic_1['industry']==1].shape[0], w2_ethnic_1[w2_ethnic_1['industry']==2].shape[0], w2_ethnic_1[w2_ethnic_1['industry']==3].shape[0], w2_ethnic_1[w2_ethnic_1['industry']==4].shape[0], w2_ethnic_1[w2_ethnic_1['industry']==5].shape[0], w2_ethnic_1[w2_ethnic_1['industry']==6].shape[0], w2_ethnic_1[w2_ethnic_1['industry']==7].shape[0]]
labels = ['Leisure n Hospitality', 'Retail', 'Education' ,'Health', 'Construction', 'Manufacturing', 'Professional n Business', 'Finance']

#define Seaborn color palette to use
colors = sns.color_palette('rocket')[0:7]

#create pie chart
plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%')
plt.title('Pie chart showing Ethnicity 1\'s preference in Industry in World 2')
plt.show()

data = [w2_ethnic_2[w2_ethnic_2['industry']==0].shape[0], w2_ethnic_2[w2_ethnic_2['industry']==1].shape[0], w2_ethnic_2[w2_ethnic_2['industry']==2].shape[0], w2_ethnic_2[w2_ethnic_2['industry']==3].shape[0], w2_ethnic_2[w2_ethnic_2['industry']==4].shape[0], w2_ethnic_2[w2_ethnic_2['industry']==5].shape[0], w2_ethnic_2[w2_ethnic_2['industry']==6].shape[0], w2_ethnic_2[w2_ethnic_2['industry']==7].shape[0]]
labels = ['Leisure n Hospitality', 'Retail', 'Education' ,'Health', 'Construction', 'Manufacturing', 'Professional n Business', 'Finance']

#define Seaborn color palette to use
colors = sns.color_palette('viridis')[0:7]

#create pie chart
plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%')
plt.title('Pie chart showing Ethnicity 2\'s preference in Industry in World 2')
plt.show()

# Ethnicity 0 has equal preference in Retail, Professional n Business, and Health. And prefers Finance the least (8%). But preference is well divided over the sectors. 

# Ethnicity 1 has higher preference for Retail, Professional n Business, Education, Manufacturing, Health over Construction or Finance (9%). But still well divided. 

# Ethnicity 2 has higher preference for Retail, Education, Health, Manufacturing, Professional n Business as compared to construction or finance (8%). But still well divided. 


# *Overall, ethnicities in World 2 are better divided within the different sectors as compared to world 1.*  


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
#
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

#%%
# Violin Plots comparing ethnicities income and industry with gender as Hue.  

fig, ax = plt.subplots(2,2,figsize = (20,15))

sns.violinplot(x = 'ethnic', y = 'industry', hue = 'gender', kind = "violin", data = world1, split = True, inner = 'quartile', ax = ax[0,0])
ax[0,0].set_title('Ethnic Vs Industry - World 1', fontsize = 18)

sns.violinplot(x = 'ethnic', y = 'industry', hue = 'gender', kind = "violin", data = world2, split = True, inner = 'quartile', ax = ax[0,1])
ax[0,1].set_title('Ethnic Vs Industry - World 2', fontsize = 18)

sns.violinplot(x = 'ethnic', y = 'income00', hue = 'gender', kind = "violin", data = world1, split = True, inner = 'quartile', ax =ax[1,0])
ax[1,0].set_title('Ethnic Vs Income - World 1', fontsize = 18)

sns.violinplot(x = 'ethnic', y = 'income00', hue = 'gender', kind = "violin", data = world2, split = True, inner = 'quartile', ax = ax[1,1])
ax[1,1].set_title('Ethnic Vs Income - World 2', fontsize = 18)

# We can observe that World 2 is equally divided when it comes to earnings and industry preference within the genders in the different ethinicities. We can also make the observation that all three ethnicities have similar quartile ranges for both income industry. 


# In world 1, however, things are more divided. From the violin plots, we can clearly observe that Women of ethnicity 2 are more involved in industry sectors 5,6,and 7 as compared to women in ethnicity 1 and 0. Same can be said for men. 
# When it comes to earnings, men of ethnicity 2 earn more than men of ethnicity 0 and 1. Also, Their income is spread over a wider range, with there being the highest number of ethnicity 3 men earning high wages. 

# %%
# World 1: 


sns.boxenplot(x = 'ethnic', y = 'education', hue = 'gender', data = world1, palette='husl')
plt.title('Ethnic vs Education years - World 1', fontsize = 20)
plt.show()

sns.catplot(x = 'ethnic', y = 'industry', hue = 'gender', kind = "boxen", col = 'marital', data = world1, palette='YlOrBr')
plt.show()

sns.catplot(x = 'ethnic', y = 'income00', hue = 'gender', kind = "boxen", col = 'marital', data = world1, palette='jet_r')
plt.show()

#%%

# World 2: 

sns.boxenplot(x = 'ethnic', y = 'education', hue = 'gender', data = world2, palette='magma_r')
plt.title('Ethnic vs Education years - World 2', fontsize = 20)

sns.catplot(x = 'ethnic', y = 'industry', hue = 'gender', kind = "boxen", col = 'marital', data = world2, palette='autumn_r')


sns.catplot(x = 'ethnic', y = 'income00', hue = 'gender', kind = "boxen", col = 'marital', data = world2, palette='viridis_r')

plt.show()

#%% [markdown]
#----------------------------------------------------------------------
# # Summary and Conclusion: 
# 
# In conclusion of all the analysis and findings from the plots and graphs we can clearly observe that World 2 is more equally divided or more equally set up when it comes to socio-economic factors like income, industry preference, education years within the different ethnicities, genders etc. 
# 
# World 2 tends to lean more towards our definition of Utopia or a fair world, when compared to World 1. To make a choice, I would consider World 2 utopian.  
#
# 
# *Below findings stand out*:
#
#  0. Overall population and other variables like marital status, ethinicity, age and education are very identical between the two worlds. 
#  1. Different ethnicities are equally divided in terms of population in both the worlds! No one ethnicity is dominant in either world. 
#  2. Average Age and Education between ethnicities is same for all three ethnicities in both the worlds!
#  3. In World 1, It can be seen that ethnicity 0 has more people working in Health, Retail and Education sectors than manufacturing or Finance.
# Ethnicity 1 has more people working in Hospitality, Health, Retail and Construction sectors. And very little in Manufacturing or Finance. 
# Ethnicity 2 has majority people working in Professional n Business, Manufacturing and Finance, and just 1% people working in the Construction sector.  
#  4. Overall, ethnicities in World 2 are better divided within the different sectors as compared to world 1. 
# Ethnicity 0 has equal preference in Retail, Professional n Business, and Health. And prefers Finance the least (8%). But preference is well divided over the sectors. 
# Ethnicity 1 has higher preference for Retail, Professional n Business, Education, Manufacturing, Health over Construction or Finance (9%). But still well divided. 
# Ethnicity 2 has higher preference for Retail, Education, Health, Manufacturing, Professional n Business as compared to construction or finance (8%). But still well divided. 
#  5. Moreover, we can observe that World 2 is equally divided when it comes to earnings and industry preference within the genders in the different ethinicities as well! We can also make the observation that all three ethnicities have similar quartile ranges in the violin plots for both income and industry. 
# In world 1, however, things are more divided. From the violin plots, we can clearly observe that Women of ethnicity 2 are more involved in industry sectors 5,6,and 7 as compared to women in ethnicity 1 and 0. Same can be said for men. 
# When it comes to earnings, men of ethnicity 2 earn more than men of ethnicity 0 and 1. Also, Their income is spread over a wider range. 
#  6. Within the genders in World 1, it can be noted that 'Construction', 'Finance' and 'Professional n business' is very uncommon among women and mostly a male dominated sector. Retail, Education and Health are very popular among women, being one of the bigger employers, as compared to men. 
#  7. Employment between men and women is more equally distributed as compared to world 1. Even in businesses like contruction, women are equally employed in world 2, with Retail, Education and Health sectors  being very good souces of employment for men in world 2 contrary to world 1. 
#  8. Overall marital status compared between the two worlds is extremely identical. In world 1 there are some men who are still unmarried or divorced compared to world 2.
#  9. From the categorical Boxen plots for World 2, we can clearly observe that the education years, industry preference and mean income is similar for all marital groups belonging to different ethnicities. However, For World 1, different ethnicities have variations in those variables. 
#
#
# As a result, *World 2* tends to lean more towards our definition of a Utopian or a fair society, which is 'Literacy', 'Racial Equality' and 'Economically Stable Job Market'.
#
#----------------------------------------------------------------------
#%%

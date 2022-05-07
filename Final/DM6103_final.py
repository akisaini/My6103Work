# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

# %%
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from matplotlib import axis
import numpy as np
import pandas as pd
import dm6103 as dm

world1 = dm.api_dsLand('World1', 'id')
world2 = dm.api_dsLand('World2', 'id')

print("\nReady to continue.")

# %% [markdown]
# # Two Worlds (Continuation from midterm: Part I - 25%)
#
# In the (midterm) mini-project, we used statistical tests and visualization to
# studied these two worlds. Now let us use the modeling techniques we now know
# to give it another try.
#
# Use appropriate models that we learned in this class or elsewhere,
# elucidate what these two world looks like.
#
# Having an accurate model (or not) however does not tell us if the worlds are
# utopia or not. Is it possible to connect these concepts together? (Try something called
# "feature importance"?)
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
# %%
# Random Forrest Classifier:
# Let us create our first model using random forest classifier with ethnic as the target variable.
#from xgboost import XGBClassifier
# For world 1:
# region
X = world1.drop(['ethnic'], axis=1)
y = world1['ethnic']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=15)
# endregion
model1 = RandomForestClassifier()
model1.fit(X_train, y_train)
model1_y_pred = model1.predict(X_test)
model1_cm = confusion_matrix(y_test, model1_y_pred)
print(model1_cm)
cr_1 = classification_report(y_test, model1_y_pred)
print(cr_1)

# Support Vector Classifier:
model2 = SVC()
model2.fit(X_train, y_train)
model2_y_pred = model2.predict(X_test)
model2_cm = confusion_matrix(y_test, model2_y_pred)
print(model2_cm)
cr_2 = classification_report(y_test, model2_y_pred)
print(cr_2)

# Classification tree:
model3 = DecisionTreeClassifier()
model3.fit(X_train, y_train)
model3_y_pred = model3.predict(X_test)
model3_cm = confusion_matrix(y_test, model3_y_pred)
print(model3_cm)
cr_3 = classification_report(y_test, model3_y_pred)
print(cr_3)

# All three models perform similarly with f1 score and accuracy in the range of 40% to 45%.
#
# This signifies that the ebove models are only 40-45% accurate in predicting the ethnicity of a person based on information like income, age, gender, education years and industry.
#
#
# Feature Importance:
# Let us use feature importance to check what variables had more variation towards the target variable:
importance = model1.feature_importances_
importance = np.sort(importance)
# summarize feature importance

imp_df = pd.DataFrame({'Feature': X.columns, 'Score': importance})
print(imp_df)
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()
#
# From the values and the plot, we can observe that income and industry are the better predictors for ethnicity.
#
# This connects with our findings in the mini-project which clearly saw the division in industry between different ethnicities in world 1.
#
# %%
# For World 2:

X = world2.drop(['ethnic'], axis=1)
y = world2['ethnic']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=15)

model1_w2 = RandomForestClassifier()
model1_w2.fit(X_train, y_train)
model1_w2_y_pred = model1_w2.predict(X_test)
model1_w2_cm = confusion_matrix(y_test, model1_w2_y_pred)
print(model1_cm)
cr_1_w2 = classification_report(y_test, model1_w2_y_pred)
print(cr_1_w2)

# Support Vector Classifier:
model2_w2 = SVC()
model2_w2.fit(X_train, y_train)
model2_w2_y_pred = model2_w2.predict(X_test)
model2_w2_cm = confusion_matrix(y_test, model2_w2_y_pred)
print(model2_w2_cm)
cr_2_w2 = classification_report(y_test, model2_w2_y_pred)
print(cr_2_w2)

# Classification tree:
model3_w2 = DecisionTreeClassifier()
model3_w2.fit(X_train, y_train)
model3_w2_y_pred = model3_w2.predict(X_test)
model3_w2_cm = confusion_matrix(y_test, model3_w2_y_pred)
print(model3_w2_cm)
cr_3_w2 = classification_report(y_test, model3_w2_y_pred)
print(cr_3_w2)

# All three models perform similarly with f1 score and accuracy between 30 and 35%.
#
# This signifies that the ebove models are 30-35% accurate in predicting the ethnicity of a person based on information like income, age, gender, education years and industry.
#
#
# Feature Importance:
# Let us use feature importance to check what variables had more variation towards the target variable:
importance = model1_w2.feature_importances_
importance = np.sort(importance)
# summarize feature importance

imp_df_w2 = pd.DataFrame({'Feature': X.columns, 'Score': importance})
print(imp_df_w2)
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()
#
# From the values and the plot, we can observe that income and industry are the better predictors for ethnicity even for world 2.
#
#
# For both the worlds feature importance suggests that income and industry are the better predictors to the target variable 'ethnic'. However, with prior information from the mini-project visualizations, we can judge that world 1 is more divided as industry and income have the highest importance values and we know that world 1 had unequal distribution of income and industry within ethnicities.
#
# Just with these models we cannot judge which world is utopia or not, as feature importance for both the worlds, even world 2 which is fairly and equally divided between ethnicities, suggests income and industry as the highest predictors.
# Parallel information like plots and graphs can help process the information better and even form a connection with the model that we can understand.


# %% [markdown]
#
# # Free Worlds (Continuation from midterm: Part II - 25%)
#
# To-do: Complete the method/function predictFinalIncome towards the end of this Part II codes.
#
# The worlds are gifted with freedom. Sort of.
# I have a model built for them. It predicts their MONTHLY income/earning growth,
# base on the characteristics of the individual. You task is to first examine and
# understand the model. If you don't like it, build you own world and own model.
# For now, please help me finish the last piece.
#
# My model will predict what is the growth factor for each person in the immediate month ahead.
# Along the same line, it also calculate what is the expected (average) salary after 1 month with
# that growth rate. You need to help make it complete, by producing a method/function that will
# calculate what is the salary after n months. (Method: predictFinalIncome )
#
# That's all. Then try this model on people like Plato, and also create some of your favorite
# people with all sort of different demographics, and see what their growth rates / growth factors
# are in my worlds. Use the sample codes after the class definition below.
#
# %%
class Person:
    """ 
    a person with properties in the utopia 
    """

    def __init__(self, personinfo):
        # age at creation or record. Do not change.
        self.age00 = personinfo['age']
        self.age = personinfo['age']  # age at current time.
        # income at creation or record. Do not change.
        self.income00 = personinfo['income']
        self.income = personinfo['income']  # income at current time.
        self.education = personinfo['education']
        self.gender = personinfo['gender']
        self.marital = personinfo['marital']
        self.ethnic = personinfo['ethnic']
        self.industry = personinfo['industry']
        # self.update({'age00': self.age00,
        #         'age': self.age,
        #         'education': self.education,
        #         'gender': self.gender,
        #         'ethnic': self.ethnic,
        #         'marital': self.marital,
        #         'industry': self.industry,
        #         'income00': self.income00,
        #         'income': self.income})
        return

    def update(self, updateinfo):
        for key, val in updateinfo.items():
            if key in self.__dict__:
                self.__dict__[key] = val
        return

    # this will allow both person.gender or person["gender"] to access the data
    def __getitem__(self, item):
        return self.__dict__[item]


# %%
class myModel:
    """
    The earning growth model for individuals in the utopia. 
    This is a simplified version of what a model could look like, at least on how to calculate predicted values.
    """

    # ######## CONSTRUCTOR  #########
    def __init__(self, bias):
        """
        :param bias: we will use this potential bias to explore different scenarios to the functions of gender and ethnicity

        :param b_0: the intercept of the model. This is like the null model. Or the current average value. 

        :param b_age: (not really a param. it's more a function/method) if the model prediction of the target is linearly proportional to age, this would the constant coefficient. In general, this does not have to be a constant, and age does not even have to be numerical. So we will treat this b_age as a function to convert the value (numerical or not) of age into a final value to be combined with b_0 and the others 

        :param b_education: similar. 

        :param b_gender: similar

        :param b_marital: these categorical (coded into numeric) levels would have highly non-linear relationship, which we typically use seaparate constants to capture their effects. But they are all recorded in this one function b_martial

        :param b_ethnic: similar

        :param b_industry: similar

        :param b_income: similar. Does higher salary have higher income or lower income growth rate as lower salary earners?
        """

        self.bias = bias  # bias is a dictionary with info to set bias on the gender function and the ethnic function

        # ##################################################
        # The inner workings of the model below:           #
        # ##################################################

        # 0.23% MONTHLY grwoth rate as the baseline. We will add/subtract from here
        self.b_0 = 0.0023

        # Technically, this is the end of the constructor. Don't change the indent

    # The rest of the "coefficients" b_1, b_2, etc are now disguised as functions/methods
    def b_age(self, age):  # a small negative effect on monthly growth rate before age 45, and slight positive after 45
        effect = -0.00035 if (age < 40) else 0.00035 if (age >
                                                         50) else 0.00007*(age-45)
        return effect

    def b_education(self, education):
        effect = -0.0006 if (education < 8) else -0.00025 if (education < 13) else 0.00018 if (
            education < 17) else 0.00045 if (education < 20) else 0.0009
        return effect

    def b_gender(self, gender):
        effect = 0
        biasfactor = 1 if (self.bias["gender"] == True or self.bias["gender"] > 0) else 0 if (
            self.bias["gender"] == False or self.bias["gender"] == 0) else -1  # for bias, no-bias, and reverse bias
        # This amount to about 1% difference annually
        effect = -0.00045 if (gender < 1) else 0.00045
        return biasfactor * effect

    def b_marital(self, marital):
        effect = 0  # let's assume martial status does not affect income growth rate
        return effect

    def b_ethnic(self, ethnic):
        effect = 0
        biasfactor = 1 if (self.bias["ethnic"] == True or self.bias["ethnic"] > 0) else 0 if (
            self.bias["ethnic"] == False or self.bias["ethnic"] == 0) else -1  # for bias, no-bias, and reverse bias
        effect = -0.0006 if (ethnic < 1) else - \
            0.00027 if (ethnic < 2) else 0.00045
        return biasfactor * effect

    def b_industry(self, industry):
        effect = 0 if (industry < 2) else 0.00018 if (industry < 4) else 0.00045 if (
            industry < 5) else 0.00027 if (industry < 6) else 0.00045 if (industry < 7) else 0.00055
        return effect

    def b_income(self, income):
        # This is the kicker!
        # More disposable income allow people to invest (stocks, real estate, bitcoin). Average gives them 6-10% annual return.
        # Let us be conservative, and give them 0.6% return annually on their total income. So say roughly 0.0005 each month.
        # You can turn off this effect and compare the difference if you like. Comment in-or-out the next two lines to do that.
        # effect = 0
        effect = 0 if (income < 50000) else 0.0001 if (income < 65000) else 0.00018 if (
            income < 90000) else 0.00035 if (income < 120000) else 0.00045
        # Notice that this is his/her income affecting his/her future income. It's exponential in natural.
        return effect
    
    
    

        # ##################################################
        # end of black box / inner structure of the model  #
        # ##################################################

    # other methods/functions
    def predictGrowthFactor(self, person):  # this is the MONTHLY growth FACTOR
        factor = 1 + self.b_0 + self.b_age(person["age"]) + self.b_education(person['education']) + self.b_ethnic(person['ethnic']) + self.b_gender(
            person['gender']) + self.b_income(person['income']) + self.b_industry(person['industry']) + self.b_marital(['marital'])
        # becareful that age00 and income00 are the values of the initial record of the dataset/dataframe.
        # After some time, these two values might have changed. We should use the current values
        # for age and income in these calculations.
        return factor

    # perdict the new income one MONTH later. (At least on average, each month the income grows.)
    def predictIncome(self, person):
        return person['income']*self.predictGrowthFactor(person)

    def predictFinalIncome(self, n, person):
        # predict final income after n months from the initial record.
        # the right codes should be no longer than a few lines.
        # If possible, please also consider the fact that the person is getting older by the month.
        # The variable age value keeps changing as we progress with the future prediction.
        #
        # This is similar to a compound intrest problem where the growth factor can be treated as the intrest amount and income as the principal amt. Intrest for the first year is x. For the nth year is x**n
        #
        #f = self.predictGrowthFactor(person) ** n
        # return the income level after n months.
        final_income =  person['income'] * (self.predictGrowthFactor(person))**(n)
        # Check for increase in years and update. 
        if n >= 12:
          count = (n/12)
          person.update( { "age": person['age00']+count, "education": person['education'], "marital":person['marital'], "income": final_income } )
        
        return final_income
    
      
print("\nReady to continue.")



# SAMPLE CODES to try out the model
utopModel = myModel({"gender": False, "ethnic": False})  # no bias Utopia model
# bias, flawed, real world model
biasModel = myModel({"gender": True, "ethnic": True})

print("\nReady to continue.")

# %%
# Now try the two models on some versions of different people.
# See what kind of range you can get. Plato is here for you as an example.
# industry: 0-leisure n hospitality, 1-retail , 2- Education 17024, 3-Health, 4-construction, 5-manufacturing, 6-professional n business, 7-finance
# gender: 0-female, 1-male
# marital: 0-never, 1-married, 2-divorced, 3-widowed
# ethnic: 0, 1, 2
# age: 30-60, although there is no hard limit what you put in here.
# income: no real limit here.

months = [1,12,60,120,360]  # Try months = 1, 12, 60, 120, 360 
# months are time starting from original age (age00)
# In the ideal world model with no bias
plato = Person({"age": 58, "education": 20, "gender": 1,
               "marital": 0, "ethnic": 2, "industry": 7, "income": 100000})
# This is the current growth factor for plato
print(f'utop: {utopModel.predictGrowthFactor(plato)}')
# This is the income after 1 month
print(f'utop: {utopModel.predictIncome(plato)}')
# Do the following line when your new function predictFinalIncome is ready
for i in months:
  print(f'utop: {utopModel.predictFinalIncome(i,plato)}')
#
# If plato ever gets a raise, or get older, you can update the info with a dictionary:
# plato.update( { "age": 59, "education": 21, "marital": 1, "income": 130000 } )

# Trying Utop model: 
# month = 1 : 100445.0 - income growth in first month
# month 12 : 105472.65471455663 - income growth in 12 months. Age: 59
# month 60 : 137669.98441393656 - income growth oin 60 months. Age: 63
# month 120 : 237370.2145143558 - income growth in 120 months. Age: 68
# month 360 : 1216710.0612302066 - income growth in 360 months. Age: 88

# %%
months = [1,12,60,120,360] 
# In the flawed world model with biases on gender and ethnicity
aristotle = Person({"age": 58, "education": 20, "gender": 1,
                   "marital": 0, "ethnic": 2, "industry": 7, "income": 100000})
# This is the current growth factor for aristotle
print(f'bias: {biasModel.predictGrowthFactor(aristotle)}')
# This is the income after 1 month
print(f'bias: {biasModel.predictIncome(aristotle)}')
# Do the following line when your new function predictFinalIncome is ready
for i in months:
 print(f'bias: {biasModel.predictFinalIncome(i,aristotle)}')
 
 
# Trying on Plato in Biased world:

plato_b = Person({"age": 58, "education": 20, "gender": 1,
               "marital": 0, "ethnic": 2, "industry": 7, "income": 100000})
# This is the current growth factor for Plato in Biased world
print(f'bias: {biasModel.predictGrowthFactor(plato_b)}')
# This is the income after 1 month
print(f'bias: {biasModel.predictIncome(plato_b)}')
# Do the following line when your new function predictFinalIncome is ready
for i in months:
 print(f'bias: {biasModel.predictFinalIncome(i,plato_b)}')

print("\nReady to continue.")

# Trying Biased model: (Aristotle)
# month = 1 : 100535.00000000001 - income growth in first month
# month 12 : 106612.31827030999 - income growth in 12 months. Age: 59
# month 60 : 146839.98205686538 - income growth oin 60 months. Age: 63
# month 120 : 281904.32356338145 - income growth in 120 months. Age: 68
# month 360 : 1994683.4231702418 - income growth in 360 months. Age: 88

# Trying Biased model: (Plato)
# month = 1 : 100535.00000000001 - income growth in first month
# month 12 : 106612.31827030999 - income growth in 12 months. Age: 59
# month 60 : 146839.98205686538 - income growth oin 60 months. Age: 63
# month 120 : 281904.32356338145 - income growth in 120 months. Age: 68
# month 360 : 1994683.4231702418 - income growth in 360 months. Age: 88


# In the biased model Plato would earn nearly 777,973 more with the same credentials due to gender and ethnic bias. 

# %% [markdown]
# # Evolution (Part III - 25%)
#
# We want to let the 24k people in WORLD#2 to evolve, for 360 months. You can either loop them through, and
# create a new income or incomeFinal variable in the dataframe to store the new income level after 30 years. Or if you can figure out a way to do
# broadcasting the predict function on the entire dataframem that can work too. If you loop through them, you can also consider
# using Person class to instantiate the person and do the calcuations that way, then destroy it when done to save memory and resources.
# If the person has life changes, it's much easier to handle it that way, then just tranforming the dataframe directly.
#
# We have just this one goal, to see what the world look like after 30 years, according to the two models (utopModel and biasModel).
#
# Remember that in the midterm, world1 in terms of gender and ethnic groups,
# there were not much bias. Now if we let the world to evolve under the
# utopia model utopmodel, and the biased model biasmodel, what will the income distributions
# look like after 30 years?
#
# Answer this in terms of distribution of income only. I don't care about
# other utopian measures in this question here.

# In utopian setting world2 would look like:
world2['income'] = world2['income00']
world2['age'] = world2['age00']


world2['incomefinal_utop'] = world2.apply(
    lambda x: utopModel.predictFinalIncome(360, x), axis=1)


# In Biased setting world2 would look like: 
world2['incomefinal_bias'] = world2.apply(
    lambda x: biasModel.predictFinalIncome(360, x), axis=1)


print(world2)
# %%
# # Reverse Action (Part IV - 25%)
#
# Now let us turn our attension to World 1, which you should have found in the midterm that
# it is far from being fair from income perspective among gender and ethnic considerations.
#
# Let us now put in place some policy action to reverse course, and create a revser bias model:
# revsered bias, to right what is wronged gradually.
revbiasModel = myModel({"gender": -1, "ethnic": -1})

# If we start off with Word 1 on this revbiasModel, is there a chance for the world to eventual become fair like World #2? If so, how long does it take, to be fair for the different genders? How long for the different ethnic groups?

# If the current model cannot get the job done, feel free to tweak the model with more aggressive intervention to change the growth rate percentages on gender and ethnicity to make it work.

#World1
# We will try to compare world1 stats 30 years or 360 months down the line with world2. 
world1['income'] = world1['income00']
world1['age'] = world1['age00']

months =  360
# months are time starting from original age (age00)
# In the reverse world bias model/ world1:
world1['incomefinal'] = world1.apply(
    lambda x: revbiasModel.predictFinalIncome(months, x), axis=1)

print(world1)
#%%
# lets us compare the different models in the two worlds now: 
import seaborn as sns
import matplotlib.pyplot as plt

#rev bias model
sns.catplot(data = world1, x = 'ethnic' , y = 'incomefinal', hue = 'gender', kind = 'boxen', palette='autumn_r')
plt.ticklabel_format(style='plain', axis='y')
plt.title('Rev Bias Model on World 1')

# From the categorical plots we can observe that there is some difference between the two genders(F/M) belonging to the three ethnicities. With females of ethnicity 0 earning the highest amount. 

#bias model
sns.catplot(data = world2, x = 'ethnic' , y = 'incomefinal_bias', hue = 'gender', kind = 'boxen', palette = 'jet_r')
plt.ticklabel_format(style='plain', axis='y')
plt.title('Bias Model on World 2')
# From the categorical plots we can observe that there is clear difference between the genders of the 3 ethnicities. This makes sense since this is the bias model where gender and ethnicity have a high bias. Men of ethnicity 2 earn the highest amount. 


#utop model
sns.catplot(data = world2, x = 'ethnic' , y = 'incomefinal_utop', hue = 'gender', kind = 'boxen', palette = 'husl')
plt.ticklabel_format(style='plain', axis='y')
plt.title('Utop Model on World 2')
# From the categorical plots we can observe that both the genders belonging to the three ethnicities earn about the same with minute or very little differences due to outliers. This makes sense since this is the utop model where everything is balanced. 

plt.show()

# %%
# Is there a chance for world1 to become like utop world 2? 
#
# 30 years in the future, world1 is definitely not like the utop world 2. 
# We can try to add another column in the world1 data set that stores the income level after 60 years or 720 months. 
# Then we can compare that scenario with the present utop world 2. 

#
# months are time starting from original age (age00)
# In the reverse world bias model/ world1:
months = 720
world1['incomefinal_60y'] = world1.apply(
    lambda x: revbiasModel.predictFinalIncome(months, x), axis=1)

print(world1)

# %%
# Plotting both boxen and violin plots for rev bias model 60 years ahead:
sns.catplot(data = world1, x = 'ethnic' , y = 'incomefinal_60y', hue = 'gender', kind = 'boxen', palette='rocket')
plt.ticklabel_format(style='plain', axis='y')
plt.title('Forecast - Rev Bias Model on World1 720 months')

sns.catplot(data = world1, x = 'ethnic' , y = 'incomefinal_60y', hue = 'gender', kind = 'violin', palette='Paired')
plt.ticklabel_format(style='plain', axis='y')

# We can observe from both the plots that although people earn a high salary, which is expected due to the growth factor being multiplied for 60 years, there are still differences between the two genders and the ethnicities. Female of ethnicity 0 still earn the highest amount. 

# To notice more prominent changes, the current reverse bias model needs to be tweaked by changing the growth rate percentages of ethnicities and gender. 

# We will try that by substituting other values in the functions and will regenrate the plots. 

# %%
tweakedmodel = myModel({'gender': -0.6, 'ethnic': -0.8})

months = 360
world1['incomefinal_tweaked'] = world1.apply(
    lambda x: tweakedmodel.predictFinalIncome(months, x), axis=1)

print(world1)

sns.catplot(data = world1, x = 'ethnic' , y = 'incomefinal_tweaked', hue = 'gender', kind = 'boxen', palette='Set2')
plt.ticklabel_format(style='plain', axis='y')
plt.title('Tweaked Model on World1')

sns.catplot(data = world1, x = 'ethnic' , y = 'incomefinal_tweaked', hue = 'gender', kind = 'violin', palette='Spectral')
plt.ticklabel_format(style='plain', axis='y')

# By changing the growth rate percentages on gender and enthnicity, we can observe that world1 undergoes a massive change. 
#
# By creating a tweaked model, with gender set to -0.6 and ethnic set to -0.8 we can observe from the plot that differences between genders in all three ethnicities have reduced to a very minimal level, with both F/M having nearly the same average income over the next 30 years or 360 months. 
#
# The overall differences between the ethnicities has also reduced greatly. This can be observed from both, boxen and violin plots. 
# 

# %%
# Checking the tweaked model 720 months or 60 years down the line on world1. 

world1['incomefinal_tweaked'] = world1.apply(
    lambda x: tweakedmodel.predictFinalIncome(720, x), axis=1)

print(world1)

sns.catplot(data = world1, x = 'ethnic' , y = 'incomefinal_tweaked', hue = 'gender', kind = 'boxen', palette='gnuplot')
plt.ticklabel_format(style='plain', axis='y')
plt.title('Forecast - Tweaked Model on World1 720 months')

# We can observe that gender differences across ethnicities have come up again over a time difference of 60 years even when using the tweaked model. 
# 
# Differences between the three ethnicities continue to grow over 60 years. Maybe these observed differences could be due to other factors also affecting the bias like industry, age, education, marital, and income in world1. 
#
# Hence, within the next 60 years, there is almost no chance for world#1 to become like utop world2. 


# %%

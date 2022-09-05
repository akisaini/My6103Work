#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%%
df = pd.read_excel('CO2_1970-2015_dataset_of_CO2_report_2016.xls', header = [0], index_col = 0, squeeze = True)
# %%
co2_ml = df.loc['Malaysia']
co2_fr = df.loc['France']
# %%
#array of numbers
year = np.arange(1970,2016)
# %%
plt.plot(year, co2_ml)
plt.plot(year,co2_fr)
plt.xlabel('Year')
plt.ylabel('CO2 level')
plt.legend(['Malaysia', 'France'], loc = 'lower right')
plt.show()
# %%
#ARIMA Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARMA
# %%
df = pd.read_csv('HCLTECH.csv')
# %%
df = df.dropna() 
# %%
df['Prev Close'] 
# %%
#Converting date column into index. 
df.index = pd.to_datetime(df['Date'])
# %%
#Just keeping 2013 Prev Close data from the entire data set for calculations. 
df = df['Prev Close']['2013-01-01':'2013-12-2']
# %%
df.describe()
#count     230.000000
#mean      852.953478
#std       156.484472
#min       618.700000
#25%       736.350000
#50%       777.450000
#75%      1023.962500
#max      1161.150000
# %%
#plotting the figure - Data Exploration
plt.plot(df)
plt.xlabel('Time Frame')
plt.ylabel('Stock Price - HCLTECH')
# %%
#Checking stationarity - Stationary means that mean and variance are constant over a time period. 
# Method 1 - Rolling Statistics
# Method 2 - Duckey Fuller (AD-Fuller Test)
roll_mean = df.rolling(12).mean()
roll_std = df.rolling(12).std()
# %%
plt.plot(df, color = 'blue', label = 'Original')
plt.plot(roll_mean, color ='red', label = 'Rolling Mean')
plt.plot(roll_std, color = 'black', label = 'Rolling Std')
plt.legend(loc = 'best') # place a legend on the plot
plt.title('Rolling Mean and Std')
plt.show()
#We see an upward trend - Non Stationary
# %%
#Making series stationary
#Taking log transformation. 
#We can take many transformations to make the series stationary - log, square root tranformation and cubed root transformation. 
tf_log = np.log(df)
plt.plot(tf_log)
#%%
#Decomposition - gives different components of the time series plot. 
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(tf_log, period = 1, model = 'multiplicative')
#%%
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# %%
plt.subplot(411)
plt.plot(tf_log, label ='original')
plt.legend(loc = 'best')
plt.subplot(412)
plt.plot(trend, label = 'Trend')
plt.legend(loc = 'best')
plt.subplot(413)
plt.plot(seasonal, label = 'Seasonality')
plt.legend(loc = 'best')
plt.subplot(414)
plt.plot(residual, label = 'Residual')
plt.legend(loc = 'best')

#From the plots, it can be noticed that seasonality and Resid are clearly missing but there is some Trend. - Not Stationary
#%%
#We try differencing now. 
#Shift by 1. 
tf_log_diff = tf_log - tf_log.shift()
plt.plot(tf_log_diff)
# %%
roll_mean = tf_log_diff.rolling(12).mean()
roll_std = tf_log_diff.rolling(12).std()

orig = plt.plot(tf_log_diff, color = 'blue', label = 'original')
mean = plt.plot(roll_mean, color = 'red', label = 'Rolling')
std = plt.plot(roll_std, color = 'black', label = 'Rolling Std')
plt.legend(loc = 'best')
plt.title('Rolling mean and std')
plt.show()
#Now, we dont see any trend. - Stationary
#To cross validate, an AD Fuller test can be used. if p value is less than 0.05 then Stationary. 

# %%
df.sort_index(inplace = True)
# %%
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.api as sm
# %%
lag_acf = acf(tf_log_diff, nlags = 20)
lag_pacf = pacf(tf_log_diff, nlags = 20)
# %%
#These figures 
fig1 = sm.graphics.tsa.plot_acf(tf_log_diff.dropna(), lags = 40)
fig2 = sm.graphics.tsa.plot_pacf(tf_log_diff.dropna(), lags = 40)
# %%
from statsmodels.tsa.arima_model import ARIMA
# %%
model = ARIMA(tf_log, order = (2,1,2))
res_ARIMA= model.fit(disp = 1)
plt.plot(tf_log_diff)
plt.plot(res_ARIMA.fittedvalues, color = 'red')
# %%
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
# %%
df = pd.read_csv('airline.txt')
# %%
df.dropna(axis = 0, inplace = True)
# %%
df.set_index('Month', inplace = True)
# %%
from statsmodels.tsa.stattools import adfuller 

def adf_test(df):
    result = adfuller(df)
    print(f'p-val : {result[1]}')
    
# %%
adf_test(df['Thousands of Passengers'])
#p-val : 0.991880243437641
#so we will not reject the null hypothesis - Hence data is not stationary
# %%
#Make stationary - use differencing. 

df['Passengers First Difference'] = df['Thousands of Passengers']-df['Thousands of Passengers'].shift(1)

# %%
adf_test(df['Passengers First Difference'].dropna())
#p-val : 0.05421329028382468
#we will not reject null hypothesis. - Non stationary
# %%
df['Passengers Second Difference'] = df['Passengers First Difference'] - df['Passengers First Difference'].shift(1)
# %%
adf_test(df['Passengers Second Difference'].dropna())
#p-val : 2.732891850014085e-29
#We will accept the alt hypothesis. Data is stationary. 
# %%

df =  pd.read_csv('cham.txt')

df.rename(columns = {'Perrin Freres monthly champagne sales millions ?64-?72':'cham'}, inplace = True)
df['Month'] = pd.to_datetime(df['Month'])
# %%
df.set_index('Month', inplace = True)
# %%
from statsmodels.tsa.stattools import adfuller 

def adf_test(df):
    result = adfuller(df)
    print(f'p-val : {result[1]}')
# %%
adf_test(df['cham'])
#Not stationary
#P-value = 0.3639
# %%
#Transformation using Differencing
df['First diff'] = df['cham'] - df['cham'].shift(1)
adf_test(df['First diff'].dropna())
#p-val : 2.5196204473868427e-10
#Stationary
# %%
#Since data is seasonal, we will shift by 12 lags. 
#This is usually done for all seasonal data's. 
# %%
df['seasonal diff'] = df['cham'] - df['cham'].shift(12)
adf_test(df['seasonal diff'].dropna())
#p-val : 2.060579696813685e-11
#We accept alt hypothesis. Data is Stationary
# %%
#ACF and PACF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# %%
#iloc gets all values after the NaN. 
fig1 = plot_acf(df['seasonal diff'].iloc[13:], lags = 40)
fig2 = plot_pacf(df['seasonal diff'].iloc[13:], lags = 40)
# %%
# ACF vs PACF: Autocorrelation function and partial ACF. 
# ACF - correlation with it's past values. 
# MA  - correlation between the past error. 
# Integrated ~~ Differenced. 
#p q d  
# P is found using PACF plot
# q is found using ACF plot 
# d = differenced. We difference the data to get rid of seasonality and trend to make it stationary. 
# D is order: how many time we difference the data. 
# Classical method - Using ACF PACF plots. 
# new method: auto_arima function. 
#%%
import itertools  # - Interesting package. 
#%%
p = range(0,8)
q = range(0,8)
d = range(0,2)

pqd_comb = list(itertools.product(p,q,d)) # itertools.products gives all the cartesian products. 


# %%

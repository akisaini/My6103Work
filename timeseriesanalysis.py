#%%
# ARIMA  -> AR - auto regression, I - integrated, MA - moving average
from pmdarima import auto_arima
import numpy as np
import pandas as pd
# %%
df = pd.read_csv('temp.csv', index_col = 'DATE', parse_dates=True)
# the parse_dates param makes pandas treat dates as date type value instead of strings.
# %%
df.dropna(inplace=True)
# %%
df.shape
df.head()
# %%

df['AvgTemp'].plot(figsize = (12,5))
# %%
from statsmodels.tsa.stattools import adfuller
def ad_fuller(dataset):
    dftest = adfuller(dataset, autolag = 'AIC')
    print(f'1. ADF:{dftest[0]}')
    print(f'2. P-vals:{dftest[1]}')
    print(f'3. Number of Lags:{dftest[2]}')
    print(f'4. Number of obs:{dftest[3]}')
    print(f'5. Critical values:{dftest[4]}')
# %%
ad_fuller(df['AvgTemp'])
# p-value is important here. It needs to be below 0.05 to accept alt hypothesis. Lower p value suggests stationarity. - constant mean and constant variance. Null hypothesis means no stationarity. 

# %%
stepwise_fit = auto_arima(df['AvgTemp'], trace = True, suppress_warnings= True)

# %%
stepwise_fit.summary()
# %%
from statsmodels.tsa.arima_model import ARIMA

train = df[:-30]
test = df[-30:]
# %%
model = ARIMA(train['AvgTemp'], order = (1,0,5))
model = model.fit()
model.summary()
# %%
# make predictions on test set:
start = len(train)
end = len(train)+len(test)-1
predictions = model.predict(start = start, end = end, typ = 'levels')
print(predictions)
predictions.index = df.index[start:end+1]
print(predictions)
# %%
predictions.plot(legend = True)
test['AvgTemp'].plot(legend = True)
# %%

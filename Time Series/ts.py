#%%
import pandas as pd
import pandas_datareader as pdr
from datetime import datetime
import matplotlib.pyplot as plt 
#%%
df = pdr.get_data_yahoo('TSLA')
# %%
dstart = datetime(2020, 1, 1)
dend = datetime(2021, 9, 1)

plt.figure(figsize = (12,4))
plt.plot(df['High'])
plt.xlim([dstart, dend]) # 2020 Jan 01 to 2021 Sept 01 records.
plt.ylim([0,900])
plt.xticks(rotation = 30)
plt.show()
#df['High'].plot(figsize = (12,4))
# %%
df.loc[dstart:dend]
# %%
pd.to_datetime(df['Date'])
df.set_index('Date', inplace = True)
# %%

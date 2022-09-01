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

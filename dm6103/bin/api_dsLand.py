#%%
import mysql.connector
from mysql.connector import Error
import pandas as pd

def api_dsLand(tbname, ind_col_name = ""):
  """ 
  call to api endpoint on datasci.land database to access datasets
  :param str tbname: table name that exist on the server 
  :param str ind_col_name: optional, name of index column 
  :return: pandas.Dataframe
  """
  
  df = None # set a global variable to store the dataframe
  apikey = 'K35wHcKuwXuhHTaz7zY42rCje'
  parameters = {"apikey": apikey, 'table': tbname}
  js = {'error': 'Initialize' }

  try:
    response = requests.get("https://api.datasci.land/endpt.json", params=parameters)
    js = response.json()
  except Error as e:
    print(f'Error while connecting to API {e}. Please contact the administrator.')

  if ('error' in js) : 
    print(f'Error: {js["error"]} Please contact the administrator.') # The json object will have a key named "error" if not successful
    return df
  
  # json object seems okay at this point
  try: df = pd.DataFrame(js) 
  except ValueError: print(f'Value Error while converting json into dataframe. Please contact the administrator.')
  except Error as e: print(f'Error while converting json into dataframe. {e}. Please contact the administrator.')
  
  # df seems load okay at this point
  if (ind_col_name and ind_col_name in df): df.set_index(ind_col_name, inplace=True)  # if given col_name exist, make it the index.
  
  # df is loaded from json now. Default values is object/string everywhere.
  # try to convert all possible ones to numeric
  for col in df.columns:
    try: df[col]=pd.to_numeric(df[col])
    except ValueError: pass
    except: pass

  print(f'Dataframe from DSLand API is loaded.')
  return df

# print("\nFunction api_dsLand loaded. Ready to continue.")

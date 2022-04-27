#%%
import os

def rev(filename): 
 if os.path.exists('filename'):
  with open(''.join(filename), 'r') as f:
    data = f.readlines()    
  if len(data) == 0:
    return('Done!') 
 else:
  if len(filename) == 0:
        return []  # base case
  else:
        return(filename[-1]. rev(filename[:-1])
 
# %%
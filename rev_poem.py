#%%
import os

def rev(filename): 
 if os.path.exists('filename'):
  with open(''.join(filename), 'r') as f:
    data = f.readlines()    
  if len(data) == 0:
    return('Done!') 
 else:
  if len(data) == 0:
        return []  # base case
  else:
        return data[-1] + rev(data[:-1])
 
#%%
with open(''.join('roses.txt'), 'r') as f:
    data = f.readlines()    

if len(data) == 0:
    print('')  # base case
else:
    print(data[-1] + rev(data[:-1]))
# %%
# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%% [markdown]
#
# HW Numpy 
# ## By: Akshat Saini       
# ### Date: 2/19/2022
#


#%%
# NumPy

import numpy as np

# %%
# ######  QUESTION 1      QUESTION 1      QUESTION 1   ##########
# This exercise is to test true/shallow copies, and related concepts. 
# ----------------------------------------------------------------
# 
# ######  Part 1a      Part 1a      Part 1a   ##########
# 
list2 = [ [11,12,13], [21,22,23], [31,32,33], [41,42,43] ] # two dimensional list (2-D array)  # (4,3)
nparray2 = np.array(list2)
print("nparray2:", nparray2)

# We will explain more of this indices function in class next. See notes in Class05_02_NumpyCont.py
idlabels = np.indices( (4,3) ) 
print("idlabels:", idlabels)

i,j = idlabels  # idlabels is a tuple of length 2. We'll call those i and j
nparray2b = 10*i+j+11
print("nparray2b:",nparray2b)

# 1.a) Is nparray2 and nparray2b the "same"? Use the logical "==" test and the "is" test. 
# Write your codes, 
# and describe what you find.
'''
#value equality and id test:
nparray2 is a copy of list2 by reference (view). nparray2b is i,j mutated.  
'''
nparray2 is nparray2b # False. nparray2 and nparray2b essentially point to two separate memory objects. They are logically equal but their memory id points to two separate objects. 

nparray2 == nparray2b # True. Logically the values are the same in nparray2b and nparray2, hence giving out a matrix with all 'true' values signifying equal values. 

# %%
# ######  Part 1b      Part 1b      Part 1b   ##########
# 
# 1.b) What kind of object is i, j, and idlabels? Their shapes? Data types? Strides?
# 
# write your codes here
#

#idlabels type of object - ndarray

i.dtype #- int32
i.shape #- (4,3)
i.strides #- (12,4)

j.dtype #-int32
j.shape #- (4,3)
j.strides #- (12,4)

idlabels.dtype #-int32
idlabels.shape #-(2,4,3)
idlabels.strides #-(48,12,4)
# %%
# ######  Part 1c      Part 1c      Part 1c   ##########
# 
# 1.c) If you change the value of i[0,0] to, say 8, print out the values for i and idlabels, both 
# before and after the change.
# i[0,0]
# write your codes here
# Describe what you find. Is that what you expect?
#
'''
Yes, this was expected as the copy created was through a reference and any change made in either of the variables will refelect on both. 
'''
#before change: 
i,j 

idlabels

#after change: 

i[0,0] = 8
i,j
idlabels

# Also try to change i[0] = 8. Print out the i and idlabels again.
i[0] = 8
i
idlabels

# %%
# ######  Part 1d      Part 1d      Part 1d   ##########
# 
# 1.d) Let us focus on nparray2 now. (It has the same values as nparray2b.) 
# Make a shallow copy nparray2 as nparray2c
nparray2c = nparray2.view() #shallow copy
# now change nparray2c 1,1 position to 0. Check nparray2 and nparray2c again. 

nparray2c[1][1] = 0
print(nparray2)
print(nparray2c)
# Print out the two arrays now. Is that what you expect?
''' Yes. changes on shallow copy relect on original copy as well. '''
# Also use the "==" operator and "is" operator to test the 2 arrays.
nparray2c is nparray2   
print(id(nparray2c))
print(id(nparray2))
'''False, as showllow copy has a different memory than the original copy.'''

nparray2c == nparray2    
'''True, as the logical values of both the arrays are the same. '''


#%%
# ######  Part 1e      Part 1e      Part 1e   ##########
# Let us try again this time using the intrinsic .copy() function of numpy array objects. 
nparray2 = np.array(list2) # reset the values. list2 was never changed.
nparray2c = nparray2.copy() 
# now change nparray2c 0,2 position value to -1. Check nparray2 and nparray2c again.
# Are they true copies?
nparray2c[0,2] = -1
print(nparray2c)
print(nparray2)
'''
no, they are not the same.
'''
# 
# write your codes here
# Again use the "==" operator and "is" operator to test the 2 arrays. 
#
# Since numpy can only have an array with all values of the same type, we usually 
# do not need to worry about deep levels copying. 
# 
# ######  END of QUESTION 1    ###   END of QUESTION 1   ##########

# %%
# ######  QUESTION 2      QUESTION 2      QUESTION 2   ##########
# Write NumPy code to test if two arrays are element-wise equal
# within a (standard) tolerance.
# between the pairs of arrays/lists: [1e10,1e-7] and [1.00001e10,1e-8]
# between the pairs of arrays/lists: [1e10,1e-8] and [1.00001e10,1e-9]
# between the pairs of arrays/lists: [1e10,1e-8] and [1.0001e10,1e-9]
# Try just google what function to use to test numpy arrays within a tolerance.

np.allclose([1e10,1e-7], [1.00001e10,1e-8]) #- False
np.allclose([1e10,1e-8], [1.00001e10,1e-9]) #- True
np.allclose([1e10,1e-8], [1.0001e10,1e-9]) #-False

# ######  END of QUESTION 2    ###   END of QUESTION 2   ##########


# %%
# ######  QUESTION 3      QUESTION 3      QUESTION 3   ##########
# Write NumPy code to reverse (flip) an array (first element becomes last).
x = np.arange(12, 38)
print(x)
x = x[::-1]
print(x)
# ######  END of QUESTION 3    ###   END of QUESTION 3   ##########


# %%
# ######  QUESTION 4      QUESTION 4      QUESTION 4   ##########
# First write NumPy code to create an 7x7 array with ones.
# Then change all the "inside" ones to zeros. (Leave the first 
# and last rows untouched, for all other rows, the first and last 
# values untouched.) 
# This way, when the array is finalized and printe out, it looks like 
# a square boundary with ones, and all zeros inside. 
# ----------------------------------------------------------------
ones = np.ones((7,7), dtype = int)
print('Original array:')
print(f'{ones}')
o = np.zeros((5,5), dtype =  int)
#Array with the border:
upd_ones = np.pad(o, pad_width = 1, mode = 'constant', constant_values = 1)
print('Padded array:')
print(f'{upd_ones}')

# ######  END of QUESTION 4    ###   END of QUESTION 4   ##########



# %%
# ######  QUESTION 5      QUESTION 5      QUESTION 5   ##########
# Broadcasting, Boolean arrays and Boolean indexing.
# ----------------------------------------------------------------
i=3642
myarray = np.arange(i,i+6*11).reshape(6,11)
#print(myarray)

# a) Obtain a boolean matrix of the same dimension, indicating if # the value is divisible by 7. 

print((myarray%7==0).astype(bool))

# b) Next get the list/array of those values of multiples of 7 in that original array  
lt = []

for i in range(len(myarray)):
    for j in range(len(myarray[i])):
        if(myarray[i][j]%7==0):
            lt.append(myarray[i][j])
            
print(lt)

# ######  END of QUESTION 5    ###   END of QUESTION 5   ##########


#
# The following exercises are  
# from https://www.machinelearningplus.com/python/101-numpy-exercises-python/ 
# and https://www.w3resource.com/python-exercises/numpy/index-array.php
# Complete the following tasks
# 

# ######  QUESTION 6      QUESTION 6      QUESTION 6   ##########

#%%
flatlist = list(range(1,25))
print(flatlist) 

#%%
# 6.1) create a numpy array from flatlist, call it nparray1. What is the shape of nparray1?
# remember to print the result
#
# write your codes here
#

nparray1 = np.array(flatlist)
print(nparray1)
nparray1.shape
#%%
# 6.2) reshape nparray1 into a 3x8 numpy array, call it nparray2
# remember to print the result
#
# write your codes here
#
nparray2 = nparray1.reshape(3,8)
print(nparray2)
#%%
# 6.3) swap columns 0 and 2 of nparray2, and call it nparray3
# remember to print the result
#
# write your codes here
#
nparray3 = nparray2.copy()
nparray3[:,[2,0]] = nparray3[:,[0,2]]
print(nparray3)
#%%
# 6.4) swap rows 0 and 1 of nparray3, and call it nparray4
# remember to print the result
#
# write your codes here
#
nparray4 = nparray3.copy()
print(nparray4) # before swapping
nparray4[[0,1],:] = nparray4[[1,0],:]
print(nparray4) # After swapping
#%%
# 6.5) reshape nparray4 into a 2x3x4 numpy array, call it nparray3D
# remember to print the result
#
# write your codes here
#
nparray3D = nparray4.reshape(2,3,4)
print(nparray3D)
nparray3D.shape
#%%
# 6.6) from nparray3D, create a numpy array with boolean values True/False, whether 
# the value is a multiple of three. Call this nparray5
# remember to print the result
# 
# write your codes here
#
nparray5 = (nparray3D%3==0).astype(bool)
print(nparray5)
#%%
# 6.7) from nparray5 and nparray3D, filter out the elements that are divisible 
# by 3, and save it as nparray6a. What is the shape of nparray6a?
# remember to print the result
#
# write your codes here
#
nparray6a = nparray3D[nparray3D%3==0]
#flatlist
print(nparray6a)
print(nparray6a.shape) 
#%%
# 6.8) Instead of getting a flat array structure, can you try to perform the filtering 
# in 6.7, but resulting in a numpy array the same shape as nparray3D? Say if a number 
# is divisible by 3, keep it. If not, replace by zero. Try.
# Save the result as nparray6b
# remember to print the result
# 
# write your codes here
#
nparray6b = np.zeros((2,3,4), dtype = int)

for i in range(len(nparray3D)):
    for j in range(len(nparray3D[i])):
        for k in range(len(nparray3D[i][j])):
            if nparray3D[i][j][k]%3==0:
                nparray6b[i][j][k] = (nparray3D[i][j][k])
            else: 
                pass
   
print(nparray6b)          
# ######  END of QUESTION 6    ###   END of QUESTION 6   ##########

#%%
#

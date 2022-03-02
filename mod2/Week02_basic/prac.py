#%%
#Python code to remove empty list from an existing list. 
# [5,4,3,[],7,[]] -> [5,4,3,7]

def cleanlst():
    # lt = input('Provide elements of list separated by')
    lt = "5,6,8,[]"
    lt_clean = list(lt.split(','))
    
    for i in range(len(lt_clean)):
        if lt_clean[i] == '[]':
            lt_clean.pop()

    return lt_clean

cleanlst()
#%%
import math                                                                          
def convgrade():
    grade = float(input('Enter your grade in numbers!'))
    if grade >= 93:
        final = 'A'
    elif 90<=grade<93:
        final = ['A-']
    elif 87<=grade<90:
        final = ['B+']
    elif 80<grade<87:
        final = ['B']
    
    return final

#%%
import numpy as np

'''
indices
'''
#      *
#     ***
#    *****
#   *******
#  *********

def startree(val):
    initstar = 1
    #val = 3
    for i in range(val):
        print(" "*val +"*"*initstar )
        initstar +=2        
        val -= 1

startree(5)

#%%

#python pr to print prime numbers upto a specified value. For eg: 
#if given 9, Print the first 9 prime numbers. 


def primenum(val):
    for i in range(2, val):
        for j in range(2,i+1):
            if i%j==0:
               continue
        print(i)

primenum(10)
#%%

#1,1,2,3,5,8,13,21......

def fib(val):
    if val == 0:
        return 1
    elif val == 1:
        return 1
    elif val == 2:
        return 2
    else:
        #fib(n) = fib(n-1)+ fib(n-2)
        return fib(val-1)+fib(val-2)  

def printseries(val):    
    for i in range(val):
        print(fib(i))
            
printseries(20) 

#%%
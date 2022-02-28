# Exercises from Prof Amir Jafari

#%%
# =================================================================
# Class_Ex1: 
# Write python program that converts seconds to 
# (x Hour, x min, x seconds)
# ----------------------------------------------------------------
import math

def time_conv(n): # 7500 - 2 hr 5 min 0 sec
    hours = (n / 3600) #2.0833..
    if hours%2 == 0: 
        return print(f'{hours} : hours')
    else:
        t1 = math.floor(hours) #2.0
    mins = (hours - t1) * 60 #2.0833 - 2 * 60 = 5.000009
    if mins%2==0:
        return print(f'{mins} : mins')
    else:
        t2 = (math.floor(mins)) #5
    sec = (mins - t2) * 60   #5.000009-5
    if sec%2==0:
        return print(f'{sec} : sec')
    else:
        t3 = math.floor(sec)
    return print(f'{t1} :Hour, {t2} :Min, {t3} :Sec')

time_conv(7999)


#%%
# =================================================================
# Class_Ex2: 
# Write a python program to print all the different arrangements of the
# letters A, B, and C. Each string printed is a permutation of ABC.
# ----------------------------------------------------------------
#ABC, ACB, BAC, CAB, CBA, BAC 
def perms(val):
    if len(val) == 0: #no value given, return empty list
        return []
    elif len(val) == 1: #value given is a single letter 
        return [val]
    # else 2 or more letters are given as input
    else:
        lst = []
        
        for i in range(0, len(val)): #length = 3
            p1 = val[i] #first element
            p2 = val[:i]+val[i+1:] #all ele till but not including the first ele + all ele after the first ele.
            for j in perms(p2):
                lst.append(p1+j)
        
        return lst        
        
perms('abc')  



#%%
# =================================================================
# Class_Ex3: 
# Write a python program to print all the different arrangements of the
# letters A, B, C and D. Each string printed is a permutation of ABCD.
# ----------------------------------------------------------------
def perms(val):
    if len(val) == 0: #no value given, return empty list
        return []
    elif len(val) == 1: #value given is a single letter 
        return [val]
    # else 2 or more letters are given as input
    else:
        lst = []
        
        for i in range(0, len(val)): #length = 3
            p1 = val[i] #first element
            p2 = val[:i]+val[i+1:] #all ele till but not including the first ele + all ele after the first ele.
            for j in perms(p2):
                lst.append(p1+j)
        
        return lst        
        
perms('abcd')  




#%%
# =================================================================
# Class_Ex4: 
# Suppose we wish to draw a triangular tree, and its height is provided 
# by the user, like this, for a height of 5:
#      *
#     ***
#    *****
#   *******
#  *********
# ----------------------------------------------------------------
def chrtree():
    level = int(input('Enter the height of the tree!'))
    star = 1
    
    for i in range(level):
        print(" " * (level-i) + '*' * star )
        star+=2
        
        

#%%
# =================================================================
# Class_Ex5: 
# Write python program to print prime numbers up to a specified values.
# ----------------------------------------------------------------

#starting after number 1. 
for i in range (2, 50):
    for j in range (2, i):
        if (i % j) == 0:
            continue
        print(i)
            




# =================================================================
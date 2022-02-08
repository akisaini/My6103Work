#%%
# print("Hello world!")
print("Hello World!")

#%%
# Question 1: Create a Markdown cell with the followings:
# Two paragraphs about yourself. In one of the paragraphs, give a hyperlink of a website 
# that you want us to see. Can be about yourself, or something you like.
#%%[markdown]

# #  **Introduction about myself!**  :smile: 

#I am Akshat, a first semester data science student. I have done my bachelors in computer science and 

#worked as a software engineer for close to 4 years following that. During work I realized I would like to 

#narrow my field of study in computers specific to analysis of data and machine learning. I have always been

#very passionate about the concept of intelligent machines and natural language processing. 

#

#I personally had a chance to develop a very small natural language processing application with a team

#during bachelors. It gave me an understanding of how data sets were to be created for simple pronouns, 

#nouns, things and places. We created our own small and basic data set of different words that the 

#application was trained on. [My Linkedin.](https://www.linkedin.com/in/akshat-saini/)
 

#%%
# Question 2: Create
# a list of all the class titles that you are planning to take in the data science program. 
# Have at least 6 classes, even if you are not a DS major
# Then print out the last entry in your list.

classes = ['Data Mining', 'Data Warehousing', 'Data Science', 'Machine Learning 1', 'Algorithm Design for Data Science', 'Natural Language Processing']

for element in classes:
    print(element)
    

#%%
# Question 3: After you completed question 2, you feel Intro to data mining is too stupid, so you are going 
# to replace it with Intro to Coal mining. Do that in python here.

classes = ['Data Mining', 'Data Warehousing', 'Data Science', 'Machine Learning 1', 'Algorithm Design for Data Science', 'Natural Language Processing']

for i in range(len(classes)):
    if classes[i] == 'Data Mining':
        classes[i] = 'Coal Mining'
    else:
        pass
print(classes)      

#%%
# Question 4: Before you go see your acadmic advisor, you are 
# asked to create a python dictionary of the classes you plan to take, 
# with the course number as key. Please do that. Don't forget that your advisor 
# probably doesn't like coal. And that coal mining class doesn't even have a 
# course number.

classdictionary = {6103:'Data Mining', 6102:'Data Warehousing', 6101:'Data Science', 6202:'Machine Learning 1', 6001:'Algorithm Design for Data Science', 6312:'Natural Language Processing'}

for k, v in classdictionary.items():
    print(f'{k}:{v}')

#%%
# Question 5: print out and show your advisor how many 
# classes (print out the number, not the list/dictionary) you plan 
# to take.

classdictionary = {6103:'Data Mining', 6102:'Data Warehousing', 6101:'Data Science', 6202:'Machine Learning 1', 6001:'Algorithm Design for Data Science', 6312:'Natural Language Processing'}

print(f'I plan on taking {len(classdictionary)} classes.')

#%%
# Question 6: Using loops 
# DO NOT use any datetime library in this exercise here 
# Use only basic loops
# Goal: print out the list of days (31) in Jan 2021 like this
# Sat - 2022/1/1
# Sun - 2022/1/2
# Mon - 2022/1/3
# Tue - 2022/1/4
# Wed - 2022/1/5
# Thu - 2022/1/6
# Fri - 2022/1/7
# Sat - 2022/1/8
# Sun - 2022/1/9
# Mon - 2022/1/10
# Tue - 2022/1/11
# Wed - 2022/1/12
# Thu - 2022/1/13
# ...
# You might find something like this useful, especially if you use the remainder property x%7
dayofweektuple = ('Sun','Mon','Tue','Wed','Thu','Fri','Sat') # day-of-week-tuple
n=31
for i in range(1,n+1):
    print(f'{dayofweektuple[(i-2)%7]} - 2022/1/{i}')
         


# %%[markdown]
# # Additional Exercise: 
# Choose three of the four exercises below to complete.
#%%
# =================================================================
# Class_Ex1: 
# Write python codes that converts seconds, say 257364 seconds,  to 
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
# Write a python codes to print all the different arrangements of the
# letters A, B, and C. Each string printed is a permutation of ABC.
# Hint: one way is to create three nested loops.
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
        
perms('abc')  


#%%
# =================================================================
# Class_Ex3: 
# Write a python codes to print all the different arrangements of the
# letters A, B, C and D. Each string printed is a permutation of ABCD.
# ----------------------------------------------------------------





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

chrtree()

# %%

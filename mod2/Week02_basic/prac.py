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
#Python code to remove empty list from an existing list. 
# [5,4,3,[],7,[]] -> [5,4,3,7]


def cleanlst():
    lt = input('Provide elements of list separated by ','')
    lt_clean = list(lt.split(','))
    
    for i in range(len(lt_clean)):
        if lt_clean[i] == '[]':
            lt_clean[i].pop()

    return lt_clean

cleanlst()
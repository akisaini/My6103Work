###############  HW  Functions      HW  Functions         HW  Functions       ###############
#%%
# ######  QUESTION 1   First, review Looping    ##########
# Write python codes to print out the four academic years for a typical undergrad will spend here at GW. 
# Starts with Sept 2021, ending with May 2025 (ntotal of 45 months), with printout like this:
# Sept 2021
# Oct 2021
# Nov 2021
# ...
# ...
# Apr 2025
# May 2025
# This might be helpful:
# If you consider Sept 2021 as a number 2021 + 8/12, you can continue to loop the increament easily 
# and get the desired year and month. (If the system messes up a month or two because of rounding, 
# that's okay for this exercise).
# And use this (copy and paste) 
# monthofyear = ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sept','Oct','Nov','Dec')
# to simplify your codes.


monthofyear = ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sept','Oct','Nov','Dec')
#for j in range(1,6):
for i in range(1,45):
    pt = (i-5)%12
    print(f'{monthofyear[pt]} 202')
#    if monthofyear[0] :
#      ++j
 # elif "":
 #   print(f'{monthofyear[pt]} 202{year[1]}')
 # elif "" :
 #   print(f'{monthofyear[pt]} 202{year[2]}')  


###############  Now:     Functions          Functions             Functions       ###############
# We will now continue to complete the grade record that we were working on in class.

#%%
###################################### Question 2 ###############################
# let us write a function find_grade(ntotal) 
# which will take your course ntotal (0-100), and output the letter grade (see your syllabus)
# have a habbit of putting in the docstring
'''
A+ 97%-100%	4.0
A	 93%-96%	3.9
A- 90%-92%	3.7
B+ 87%-89%	3.3
B	 83%-86%	3.0
B- 80%-82%	2.7
C+ 77%-79%	2.3
C	 73%-76%	2.0
C- 70%-72%	1.7
D+ 67%-69%	1.3
D	 63%-66%	1.0
D- 60%-62%	0.7
F	 0%-59%	  0.0
'''

total = 62.1

def find_grade(total):
  
  ntotal = float(total)
  
  if ntotal > 96 and ntotal <= 100:
    print('A+')
  elif ntotal > 92 and ntotal <= 96:
    print('A')
  elif ntotal > 89 and ntotal <= 92:
    print('A-')
  elif ntotal > 86 and ntotal <= 89:
    print('B+')
  elif ntotal > 82 and ntotal <= 86:
    print('B')
  elif ntotal > 79 and ntotal <= 82:
    print('B-')
  elif ntotal > 76 and ntotal <= 79:
    print('C+')
  elif ntotal > 72 and ntotal <= 76:
    print('C')
  elif ntotal > 69 and ntotal <= 72:
    print('C-')
  elif ntotal > 66 and ntotal <= 69:
    print('D+')
  elif ntotal > 62 and ntotal <= 66:
    print('D')
  elif ntotal > 59 and ntotal <= 62:
    print('D-')
  else :
    print('F') 

# Try:
find_grade(total)

# Also answer these: 
''' What is the input (function argument) data type for total? - Float type
    What is the output (function return) data type for find_grade(total) ? String type '''


#%%
###################################### Question 3 ###############################
# next the function to_gradepoint(grade)
# which convert a letter grade to a grade point. A is 4.0, A- is 3.7, etc
'''
A+ 97%-100%	4.0
A	 93%-96%	3.9
A- 90%-92%	3.7
B+ 87%-89%	3.3
B	 83%-86%	3.0
B- 80%-82%	2.7
C+ 77%-79%	2.3
C	 73%-76%	2.0
C- 70%-72%	1.7
D+ 67%-69%	1.3
D	 63%-66%	1.0
D- 60%-62%	0.7
F	 0%-59%	  0.0
'''

val = 'C'
def to_gradepoint(val):
  
  if val=='A+':
    print(4.0)
  elif val=='A':
    print(3.9)
  elif val=='A-':
    print(3.7)
  elif val=='B+':
    print(3.3)
  elif val=='B':
    print(3.0)
  elif val=='B-':
    print(2.7)
  elif val=='C+':
    print(2.3)
  elif val=='C':
    print(2.0)
  elif val=='C-':
    print(1.7)
  elif val=='D+':
    print(1.3)
  elif val=='D':
    print(1.0)
  elif val=='D-':
    print(0.7)  
  else:
    print(0.0)   
  

# Try:
print(to_gradepoint(val))

'''
# What is the input (function argument) data type for find_grade? - String type
# What is the output (function return) data type for find_grade(grade) ? Float type
'''


#%%
###################################### Question 4 ###############################
# next the function to_gradepoint_credit(course)
# which calculates the total weight grade points you earned in one course. Say A- with 3 credits, that's 11.1 total grade_point_credit
course = { "class":"IntroDS", "id":"DATS 6101", "semester":"spring", "year":2018, "grade":'B-', "credits":3 } 

def to_gradepoint_credit(course):
  # write an appropriate and helpful docstring
  # ??????    fill in your codes here
  # grade_point_credit = ?????
  # eventually, if you need to print out the value to 2 decimal, you can 
  # try something like this for floating point values %f
  # print(" %.2f " % grade_point_credit)
  return grade_point_credit

# Try:
print(" %.2f " % to_gradepoint_credit(course) )

# What is the input (function argument) data type for to_gradepoint_credit? 
# What is the output (function return) data type for to_gradepoint_credit(course) ?


#%%
###################################### Question 5 ###############################
# next the function gpa(courses) to calculate the GPA 
# It is acceptable syntax for list, dictionary, JSON and the likes to be spread over multiple lines.
courses = [ 
  { "class":"Intro to DS", "id":"DATS 6101", "semester":"spring", "year":2020, "grade":'B-', "credits":3 } , 
  { "class":"Data Warehousing", "id":"DATS 6102", "semester":"fall", "year":2020, "grade":'A-', "credits":4 } , 
  { "class":"Intro Data Mining", "id":"DATS 6103", "semester":"spring", "year":2020, "grade":'A', "credits":3 } ,
  { "class":"Machine Learning I", "id":"DATS 6202", "semester":"fall", "year":2020, "grade":'B+', "credits":4 } , 
  { "class":"Machine Learning II", "id":"DATS 6203", "semester":"spring", "year":2021, "grade":'A-', "credits":4 } , 
  { "class":"Visualization", "id":"DATS 6401", "semester":"spring", "year":2021, "grade":'C+', "credits":3 } , 
  { "class":"Capstone", "id":"DATS 6101", "semester":"fall", "year":2021, "grade":'A-', "credits":3 } 
  ]

def find_gpa(courses):
  # write an appropriate and helpful docstring
  ntotal_grade_point_credit =0 # initialize 
  ntotal_credits =0 # initialize
  # ??????    fill in your codes here
  # gpa = ?????
  

# Try:
print(" %.2f " % find_gpa(courses) )

# What is the input (function argument) data type for find_gpa? 
# What is the output (function return) data type for find_gpa(courses) ?


#%%
###################################### Question 6 ###############################
# Write a function to print out a grade record for a single class. 
# The return statement for such functions should be None or just blank
# while during the function call, it will display the print.
course = { "class":"IntroDS", "id":"DATS 6101", "semester":"spring", "year":2018, "grade":'B-', "credits":3 } 


def printCourseRecord(course):
  # write an appropriate and helpful docstring
  # use a single print() statement to print out a line of info as shown here
  # 2018 spring - DATS 6101 : Intro to DS (3 credits) B-  Grade point credits: 8.10 
  key = list(course.keys())
  val = list(course.values())
  print(f'{val[3]} {val[2]} - {val[1]} : {val[0]} ({val[5]} {key[5]}) {val[4]} Grade point credits: 8.10')
  return None# or return None
  
# Try:
printCourseRecord(course)

# What is the input (function argument) data type for printCourseRecord? - Dictionary type
# What is the output (function return) data type for printCourseRecord(course) ? - String type 


#%%
###################################### Question 7 ###############################
# write a function (with arguement of courses) to print out the complete transcript and the gpa at the end
# 2018 spring - DATS 6101 : Intro to DS (3 credits) B-  Grade point credits: 8.10 
# 2018 fall - DATS 6102 : Data Warehousing (4 credits) A-  Grade point credits: 14.80  
# ........  few more lines
# Cumulative GPA: ?????
 
def printTranscript(courses):
  # write an appropriate and helpful docstring
  for course in courses:
    # print out each record as before
  
  # after the completion of the loop, print out a new line with the gpa info
  
  return # or return None

# Try to run, see if it works as expected to produce the desired result
# courses is already definted in Q4
printTranscript(courses)
# The transcript should exactly look like this: 
# 2020 spring - DATS 6101 : Intro to DS (3 credits) B- Grade point credits: 8.10
# 2020 fall - DATS 6102 : Data Warehousing (4 credits) A- Grade point credits: 14.80
# 2020 spring - DATS 6103 : Intro Data Mining (3 credits) A Grade point credits: 12.00
# 2020 fall - DATS 6202 : Machine Learning I (4 credits) B+ Grade point credits: 13.20
# 2021 spring - DATS 6203 : Machine Learning II (4 credits) A- Grade point credits: 14.80
# 2021 spring - DATS 6401 : Visualization (3 credits) C+ Grade point credits: 6.90
# 2021 fall - DATS 6101 : Capstone (3 credits) A- Grade point credits: 11.10
# Cumulative GPA: 3.37

# What is the input (function argument) data type for printTranscript?  - List
# What is the output (function return) data type for printTranscript(courses) ?



#%% 
# ######  QUESTION 8   Recursive function   ##########
# Write a recursive function that calculates the Fibonancci sequence.
# The recusive relation is fib(n) = fib(n-1) + fib(n-2), 
# and the typically choice of seed values are fib(0) = 0, fib(1) = 1. 
# From there, we can build fib(2) and onwards to be 
# fib(2)=1, fib(3)=2, fib(4)=3, fib(5)=5, fib(6)=8, fib(7)=13, ...
# Let's set it up from here:

def fib(n):
  """
  Finding the Fibonacci sequence with seeds of 0 and 1
  The sequence is 0,1,1,2,3,5,8,13,..., where 
  the recursive relation is fib(n) = fib(n-1) + fib(n-2)
  :param n: the index, starting from 0
  :return: the sequence
  """
  # assume n is positive integer
  # ??????    fill in your codes here
  # n = index 
  if n == 0:
    return 0
  
  #if n is at index 1 or at index 2, return 1 - preset values. 
  elif n == 1 or n == 2:
    return 1
  
  else:
    return fib(n-1) + fib(n-2)

  

fib(20)
 





#%% 
# ######  QUESTION 9   Recursive function   ##########
# Similar to the Fibonancci sequence, let us create one (say dm_fibonancci) that has a  
# modified recusive relation dm_fibonancci(n) = dm_fibonancci(n-1) + 2* dm_fibonancci(n-2) - dm_fibonancci(n-3). 
# Pay attention to the coefficients and their signs. 
# And let us choose the seed values to be dm_fibonancci(0) = 1, dm_fibonancci(1) = 1, dm_fibonancci(2) = 2. 
# From there, we can build dm_fibonancci(3) and onwards to be 1,1,2,3,6,10,...
# Let's set it up from here:

def dm_fibonancci(n):
  """
  Finding the dm_Fibonacci sequence with seeds of 1, 1, 2 for n = 0, 1, 2 respectively
  The sequence is 0,1,1,2,3,5,8,13,..., where 
  the recursive relation is dm_fibonancci(n) = dm_fibonancci(n-1) + 2* dm_fibonancci(n-2) - dm_fibonancci(n-3)
  :param n: the index, starting from 0
  :return: the sequence
  """
  # assume n is positive integer index
  # ??????    fill in your codes here
  if n == 0 or n == 1:
    return 1 
  
  elif n == 2:
    return 2
  
  else:
    return dm_fibonancci(n-1) + 2*dm_fibonancci(n-2) - dm_fibonancci(n-3)
  
  
  
for i in range(12):
  print(dm_fibonancci(i))  # should gives 1,1,2,3,6,10,...


#%%


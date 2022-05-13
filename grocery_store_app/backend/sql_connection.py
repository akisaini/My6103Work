#%%
# Set up connection to mysqlworkbench
from __future__ import print_function
from datetime import date, datetime, timedelta
import mysql.connector

# creating a global variable to store the connection value. So if the the method is called multiple times, the connection is not created multiple time. Connection value will be stored in the global variable'__cnx' and will be fetched from there. 
__cnx = None

def get_sql_connection():
    global __cnx
    # if __cnx is none then only create it. 
    if __cnx is None: 
        __cnx = mysql.connector.connect(user='root', password='root',
                                    host='127.0.0.1',
                                    database='grocery_store')
        
    return __cnx

# %%

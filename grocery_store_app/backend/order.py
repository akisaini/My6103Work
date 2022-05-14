#%%
from sql_connection import get_sql_connection
from datetime import datetime

def insert_order(connection, order):
    cursor = connection.cursor()
    
    query = 'INSERT into orders (datetime, customer_name, total_cost) VALUES (%s, %s, %s)'
    
    order_data = (datetime.now(), order['customer_name'], order['total_cost'])
    
    cursor.execute(query, order_data)
    
    order_id = cursor.lastrowid
    
    order_details_query = 'INSERT INTO order_details (order_id, product_id, price_per_unit, quantity, total) VALUES (%s, %s, %s, %s, %s)'
    
    order_details_data = []
    
    for order_details_record in order['order_details']:
        order_details_data.append([
            # will fetch the order_id from the returned value, since for all the products the order_id has to remain the same. 
            order_id,
            # product_id has to be given by the user. 
            int(order_details_record['product_id']),
            float(order_details_record['price_per_unit']),
            float(order_details_record['quantity']),
            float(order_details_record['total'])
        ])
    
    cursor.executemany(order_details_query, order_details_data)
    
    connection.commit()
    
    return order_id

'''
if __name__ == '__main__':
    connection = get_sql_connection()
    print(insert_order(connection, 
                       {'customer_name': 'Mr. Saini',
                        'total_cost': 40,
                        'order_details':[
                            {
                                'product_id':1,
                                'price_per_unit': 10,
                                'quantity': 2,
                                'total': 20
                            },
                            {
                                'product_id': 3,
                                'price_per_unit': 10,
                                'quantity': 2,
                                'total':20                               
                            }
                        ]     
                       }
                       ))
'''                       
      
# Method to return all the orders    
    
def return_orders(connection):
    
    cursor = connection.cursor()
    
    query = 'SELECT * FROM orders;'
    
    cursor.execute(query)
    
    response = []
    for (order_id, datetime, customer_name, total_cost) in cursor:
        response.append({
            'order_id': order_id,
            'datetime': datetime,
            'customer_name': customer_name,
            'total_cost': float(total_cost)
        })
    
    return response



if __name__ == '__main__':
    connection = get_sql_connection()
    print(return_orders(connection))
#%%    


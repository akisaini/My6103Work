#%%
from sql_connection import get_sql_connection
# %%
# Get products from db

def get_all_products(connection):

    cursor = connection.cursor()

    query = 'SELECT p.product_id, p.name, p.uom_id, p.price_per_unit, u.unit_name FROM products p JOIN UOM u ON p.uom_id = u.uom_id;'

    cursor.execute(query)
    
    response = []
    
    for (product_id, name, uom_id, price_per_unit, unit_name) in cursor:
        response.append(
            {
            'product_id':product_id,
            'name':name,
            'uom_id':uom_id,
            'unit_name':unit_name,
            'price_per_unit': float(price_per_unit)
            }
            )

    return response


# if name equals main -> call some function. something that will print the products.
if __name__ == "__main__":
    connection = get_sql_connection()
    print(get_all_products(connection))

#%%
# product is a dict input by the user. 
def insert_new_product(connection, product):
    cursor = connection.cursor()
    
    query = 'INSERT INTO products (name, uom_id, price_per_unit) VALUES (%s, %s, %s);'
    data = (product['name'], product['uom_id'], product['price_per_unit'])
    cursor.execute(query, data)
    
    connection.commit()
    
    return cursor.lastrowid
    

if __name__ == "__main__":
    connection = get_sql_connection()
    print(insert_new_product(connection, {
        'name': 'cabbage',
        'uom_id': 1,
        'price_per_unit': 2
    }))
    
#%%
def delete_product(connection, product_id):
    cursor = connection.cursor()
    
    query = 'DELETE FROM products WHERE product_id = ' + str(product_id)
    cursor.execute(query) 
    connection.commit()



if __name__ == "__main__":
    connection = get_sql_connection()
    print(delete_product(connection, 16))

# %%

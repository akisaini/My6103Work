#%%
from flask import Flask, jsonify, Request
import products
from sql_connection import get_sql_connection
#%%
app = Flask(__name__)


connection = get_sql_connection()

@app.route('/getProducts', methods = ['GET'])
def get_products():
    productslist = products.get_all_products(connection)
    response = jsonify(productslist)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

'''
@app.route('/deleteProduct', methods = ['POST'])
def delete_product():
    return_id = products.delete_product(connection, Request.form['product_id'])
    response = jsonify(
        {   
            'product_id':return_id
        }
    )
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response
'''
if __name__ == '__main__':
    print('Starting python Flask Server for grocery store management')
    app.run(debug=True, use_reloader=False, port = 5000)
    
    
# %%

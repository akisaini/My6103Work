#%%
from flask import Flask, jsonify, request
import products
from sql_connection import get_sql_connection
#%%
app = Flask(__name__)

connection = get_sql_connection()

@app.route('/getProducts')
def get_products():
    products = products.get_all_products(connection)
    response = jsonify(products)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == '__main__':
    print('Starting python Flask Server for grocery store management')
    app.run(port = 5000)
    
    
# %%

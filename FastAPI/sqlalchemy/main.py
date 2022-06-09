#%%
from sqlalchemy.orm import declarative_base 
from sqlalchemy import String, Column, DateTime, Integer, create_engine 
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import nest_asyncio
nest_asyncio.apply()
from sqlalchemy import Boolean
from typing import Optional
#%%
app = FastAPI()


class Item(BaseModel): 
    id: int
    name: str
    description: str
    price: int
    on_offer: bool

@app.get('/')
async def simple_greet():
    return {'Hello': 'Akshat'}

@app.put('/item/{item_id}')
async def update_item(item_id: int, item:Item):
    return {'id': item_id,
            'name': item.name,
            'description': item.description,   
            'price': item.price,
            'on_offer':item.on_offer
            }


#%%
if __name__  == '__main__':
    uvicorn.run(app, host = '127.0.0.1', log_level = 'info')

#%%
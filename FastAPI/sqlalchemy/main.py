#%%
from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from sqlalchemy import null
import uvicorn
import nest_asyncio
nest_asyncio.apply()
from typing import Optional, List
from db import SessionLocal
import models 
#%%
app = FastAPI()


class Item(BaseModel): 
    id: int
    name: str
    description: str
    price: int
    on_offer: bool
    
    
    class Config:
        orm_mode = True

db = SessionLocal()


@app.get('/items', response_model=List[Item], status_code=200)
def get_all_items():
    items = db.query(models.Item).all()
    return items

@app.get('/item/{item_id}', response_model=Item, status_code = 200)
def get_an_item(item_id:int):
    return db.query(models.Item).filter(models.Item.id == item_id).first()

@app.post('/items', response_model=Item, status_code = status.HTTP_201_CREATED)
def create_an_item(item:Item):
    new_item = models.Item(
        name = item.name,
        description = item.description,
        price = item.price,
        on_offer = item.on_offer
    )
    
    db_item = db.query(models.Item).filter(item.name == new_item.name).first()
    
    if db_item is not NONE: 
        raise HTTPException(status_code=400, detail = 'Item exists!')
    
    db.add(new_item)
    db.commit()
    
    
    return new_item

# for updating
@app.put('/item/{item_id}', response_model=Item, status_code = status.HTTP_201_CREATED)
def update_an_item(item_id: int, item: Item):
    item_to_update = db.query(models.Item).filter(models.Item.id == item_id).first()
    item_to_update.name = item.name
    item_to_update.description = item.description
    item_to_update.price = item.price
    item_to_update.on_offer = item.on_offer
    
    db.commit()

    return item_to_update
    
# for deleting
@app.delete('/item/{item_id}', response_model=Item, status_code=status.HTTP_200_OK)
def delete_item(item_id: int):
    item_to_delete = db.query(models.Item).filter(models.Item.id == item_id).first()
    
    db.delete(item_to_delete)
    db.commit()


#%%
if __name__  == '__main__':
    uvicorn.run(app, host = '127.0.0.1', log_level = 'info')

#%%
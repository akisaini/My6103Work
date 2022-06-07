#%%
from fastapi import FastAPI
import uvicorn
import nest_asyncio
nest_asyncio.apply()
from pydantic import BaseModel
from datetime import datetime
import sqlalchemy
import databases 
import alembic

app = FastAPI()

class User(BaseModel):
    username: str
    email: str
#    created_at: datetime
#    updated_at: datetime

@app.get('/')
async def root():
    return {'message': 'Hello World'}

@app.get('/polls')
async def root():
    return {'polls': 'Hello World'}

@app.get('/users')
async def root():
    return {'users': 'Hello World'}


@app.post('/users/')
async def create_item(user: User):
    return user
    
#%%

# class poll 

'''
title
type
created_by
created_at
updated_at
is_voting_active
is_add_choices_active
'''    
class Poll(BaseModel):
    title: str
    type: str
    is_voting_active: bool
    is_add_choices_active: bool
    created_by: int
#    created_at: datetime
#    updated_at: datetime

@app.post('/polls')
async def create_poll(poll: Poll):
    return poll
    

#%%
if __name__ == '__main__':
    uvicorn.run(app, host = '127.0.0.1', port = 5000, log_level = 'info')
# %%
import alembic
# %%

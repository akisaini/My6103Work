#%%
from fastapi import FastAPI, HTTPException
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import nest_asyncio
nest_asyncio.apply()
from database import create_todo, fetch_all_todos, fetch_one_todo, remove_todo, update_todo_desc
from models import Todo

app = FastAPI()

# reactjs port
origins = ['https://localhost:3000']

# This is for the frontend/backend to communicate with eachother.
# CORS - Cross origin resource sharing.  
app.add_middleware(
    CORSMiddleware,
    allow_origins = ['*'],
    allow_credentials = True,
    allow_methods = ['*'],
    allow_headers = ['*'],
)

# basic 
@app.get('/')
async def test_conn():
    return {'Hello': 'Just a basic test'}

# fetches all todos 
@app.get('/api/todo', response_model=Todo)
async def get_todo():
    response = await fetch_all_todos()
    return response

# fetch todo by title
@app.get('/api/todo{title}', response_model=Todo)
async def get_todo_by_title(title):
    response = await fetch_one_todo(title)
    if response:
        return response
    raise HTTPException(404, f'There is no todo item with this title: {title}')

# Adds a new list given title and description.
@app.post('/api/todo', response_model= Todo)
async def add_todo(todo: Todo):
    response = await create_todo(todo.dict()) # passing a dict. 
    if response:
        return response
    raise HTTPException(400, 'Something went wrong/ Bad request')

# updates a description given a title and new desc
@app.put('/api/todo{title}', response_model= Todo)
async def update_todo(title: str, desc: str):
    response = await update_todo_desc(title, desc)
    return response 

# deletes a list given a title
@app.delete('/api/todo{title}')
async def delete_todo(title):
    response = await remove_todo(title)
    if response: 
        return 'Successfully deleted todo item!'
    raise HTTPException(404, f'There is no todo item with this title: {title}')


# %%

if __name__ == '__main__':
    uvicorn.run(app, host = 'localhost', port = 3000)
# %%

#%%
from fastapi import FastAPI, HTTPException
import uvicorn 
from models import Todo
import nest_asyncio
nest_asyncio.apply()
from database import (
    fetch_one_todo,
    fetch_all_todos,
    create_todo,
    update_todo_desc,
    remove_todo,
)

# an HTTP-specific exception class  to generate exception information

from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

origins = [
    "http://localhost:8000",
]

# CORS - cross origin resource sharing. A way for the frontend/backend to connect with eachother.  

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# basic 
@app.get('/')
async def test_conn():
    return {'Hello': 'Just a basic test'}


# fetches all todos 
@app.get("/api/todo")
async def get_todo():
    response = await fetch_all_todos()
    return response

# fetch todo by title
@app.get('/api/todo/{title}', response_model=Todo)
async def get_todo_by_title(title):
    response = await fetch_one_todo(title)
    if response:
        return response
    raise HTTPException(404, f'There is no todo item with this title: {title}')

# Adds a new list given title and description.
@app.post('/api/todo/', response_model= Todo)
async def add_todo(todo: Todo): # todo is an object of Todo. .dict() is a method in the Todo class which return a dict of the models values. 
    response = await create_todo(todo.dict()) # passing a dict. 
    if response:
        return response
    raise HTTPException(400, 'Something went wrong/ Bad request')

# updates a description given a title and new desc
@app.put('/api/todo/{title}/', response_model= Todo)
async def update_todo(title: str, desc: str):
    response = await update_todo_desc(title, desc)
    return response 

# deletes a list given a title
@app.delete('/api/todo/{title}')
async def delete_todo(title):
    response = await remove_todo(title)
    if response: 
        return 'Successfully deleted todo item!'
    raise HTTPException(404, f'There is no todo item with this title: {title}')
# %%
if __name__ == '__main__':
    uvicorn.run(app, host = 'localhost', port = 3000)
# %%

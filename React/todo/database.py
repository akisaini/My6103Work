#%%

# motor is mongodb engine to connect db with server. 
import motor.motor_asyncio
from models import Todo


client = motor.motor_asyncio.AsyncIOMotorClient('mongodb+srv://akshatsaini:Appymaladwest5656%25@cluster0.up6wz.mongodb.net/')

database = client.TodoList
collection = database.todo

# get one todo given title
async def fetch_one_todo(title):
    doc = await collection.find_one({'title': title})
    return doc

# get all todos
#async def fetch_all_todos():
    todos = []
    cursor = collection.find({})
    async for document in cursor: 
        todos.append(Todo(**document))
    return todos
 
async def fetch_all_todos():
    todos = []
    cursor = collection.find({})
    async for document in cursor:
        # Below, document(parameter) is being described as an object of class Todo. It validates that the data type of title and description is 'str'. Input has to match the defined model. For eg: data = {'title':1, 'description': 'hello there'} Here title is an int. Hence - value is not a valid string (type=type_error.string). 
        # Todo(**data) will output Todo(title='1', description='hello there'). This can furthur be converted into a dict() or JSON using - Todo(**data).dict(). 
        todos.append(Todo(**document))
    return todos

# create todo
# todo parameter is a dict document.
async def create_todo(todo):
   doc = todo
   result = await collection.insert_one(doc)
   return doc

# update todo
# '$set' is used to define updates 
async def update_todo_desc(title, desc):
    await collection.update_one({'title':title}, {'$set': {'description': desc}})
    return fetch_one_todo(title)
    
# delete todo
async def remove_todo(title):
    await collection.delete_one({'title': title})
    return True

# %%

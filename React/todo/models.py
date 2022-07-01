#%%
from pydantic import BaseModel

# main way to define objects in pydantic is through models. 
# defining Todo object below. 

class Todo(BaseModel):
    title: str
    description: str


# %%

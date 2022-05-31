# Setting up FASTapi server
#%%
from fastapi import FastAPI, UploadFile, File
import uvicorn
# Below is very important for uvicorn to work.
import nest_asyncio
nest_asyncio.apply()
#%%
app = FastAPI()
#%%
@app.get("/ping")
async def ping():
    return {"Hello": "I am on"}

@app.post('/predict')
async def predict(
    file: UploadFile = File(...)
):
    
    pass

#%%
if __name__ == '__main__':
    uvicorn.run(app, host = 'localhost')
# %%


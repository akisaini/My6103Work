# Setting up FASTapi server
#%%
from fastapi import FastAPI, UploadFile, File
import uvicorn
# Below is very important for uvicorn to work.
import nest_asyncio
nest_asyncio.apply()
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
#%%
app = FastAPI()

my_model = tf.keras.models.load_model('./tomato_cnn/saved_model/')
my_model.summary()
#%%
@app.get("/ping")
async def ping():
    return {"Hello": "I am on!"}

# '->' is python return annotation. It means the function will most likely return a np.ndarray or whatever is being mentioned but is not forced to return in that particular type. 
def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image
#%%
@app.post('/predict')
async def predict(
    # file: is the key name. it can be anything. 
    file: UploadFile = File(...)
):
    # file.read() contains raw file data. 
    # This raw data needs to be converted to numpy array of tf array. 
    img = read_file_as_image(await file.read())
    return img
#%%
if __name__ == '__main__':
    uvicorn.run(app, host = 'localhost')
# %%


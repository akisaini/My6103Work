''' import boto3'''
#%%%
import psycopg2
from typing import List
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI, status, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import nest_asyncio
nest_asyncio.apply()
from psycopg2 import Error
from db import SessionLocal
import models 


app = FastAPI(debug = True)
app.add_middleware(
    CORSMiddleware,
    allow_origins = ['*'],
    allow_credentials = True,
    allow_methods = ['*'],
    allow_headers = ['*'],
)

class VideoModel(BaseModel):
    id: int
    video_title: str
    video_url: str

    class Config:
        orm_mode = True

db = SessionLocal()


@app.get('/status')
async def check_status():
    return 'Hello'
'''
@app.get('/videos', response_model=List[VideoModel])
async def get_videos():

    # db connection
    try:
        connection = psycopg2.connect(
            database = 'vids', user = 'postgres', password = 'root', host = '127.0.0.1'
        )
    
        cursor = connection.cursor()
        
        # Executing a SQL query
        cursor.execute("SELECT * FROM vidlist;")
        # Fetch result
        rows = cursor.fetchall()
        formatted_videos = []
     
        for row in rows:
            formatted_videos.append(
            VideoModel(id = row[0], video_title=row[1], video_url=[2])
        )
        
        return formatted_videos
    
    except (Exception, Error) as error:
        print("Error while connecting to PostgreSQL", error)
    finally:
        if (connection):
            cursor.close()
            connection.close()   '''
            
@app.get('/vid', response_model=List[VideoModel])
def get_all_items():
    items = db.query(models.VideoModel).all()
    return items
      


#%%
if __name__  == '__main__':
    uvicorn.run(app, host = '127.0.0.1', log_level = 'info')

# %%
''' cur = conn.cursor()
    cur.execute('SELECT (id, video_list, video_url) FROM vidlist;')
    rows = cur.fetchall()
    
    formatted_videos = []
     
    for row in rows:
        formatted_videos.append(
            VideoModel(id = row[0], video_title=row[1], video_url=[2])
        )
         
    cur.close()
    conn.close()
    return formatted_videos'''
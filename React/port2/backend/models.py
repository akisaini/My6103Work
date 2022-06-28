from db import Base
from sqlalchemy import String, Column, DateTime, Integer, Boolean, Text

class VideoModel(Base): 
    __tablename__ = 'vidlist'
    id= Column(Integer(), primary_key = True)
    video_title= Column(Text)
    video_url= Column(Text)
    
from db import Base
from sqlalchemy import String, Column, DateTime, Integer, Boolean, Text

class Item(Base): 
    __tablename__ = 'items'
    id= Column(Integer(), primary_key = True)
    name= Column(String(255), nullable = False)
    description= Column(Text, nullable = False)
    price= Column(Integer(), nullable = False)
    on_offer= Column(Boolean(), nullable = False)
    

    
    
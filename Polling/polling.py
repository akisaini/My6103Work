#%%
from fastapi import FastAPI, Depends, HTTPException
import uvicorn
import nest_asyncio
nest_asyncio.apply()
from pydantic import BaseModel
from datetime import datetime
import databases 
from sqlalchemy import engine_from_config, pool
from sqlalchemy.orm import Session
from alembic import context 
from logging.config import fileConfig
from typing import List

from db.models.models import User, Poll 
from db.db import SessionLocal, engine, Base
#%%
Base.metadata.create_all(bind=engine)

app = FastAPI()


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



def get_user(db: Session, user_id: int):
    return db.query(User).filter(User.id == user_id).first()


def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()


def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(User).offset(skip).limit(limit).all()


class User(BaseModel):
    username: str
    email: str
    created_at: datetime 
    updated_at: datetime


class UserCreate(BaseModel):
    username: str
    email: str

def create_user(db: Session, user: UserCreate):
    db_user = User(email=user.email, username = user.username)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


@app.get('/')
async def root():
    return {'message': 'Hello World'}

@app.get('/polls')
async def root():
    return {'polls': 'Hello World'} 


@app.get("/users/", response_model=list[User])
def get_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = get_users(db, skip=skip, limit=limit)
    return users

@app.post("/users/", response_model=User)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    return create_user(db=db, user=user)
    
#%%

# class poll 

'''
title
type ( img or text)
created_by
created_at
updated_at
is_voting_active
is_add_choices_active
'''    
class Poll(BaseModel):
    title: str
    type: str
    is_voting_active: bool # can allow voting to start or not. 
    is_add_choices_active: bool # Toggle the option to allow people to add options to the poll (they can toggle off and on when they want)
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

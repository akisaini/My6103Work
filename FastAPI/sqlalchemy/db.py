#%%
from sqlalchemy.engine import create
from sqlalchemy.orm import declarative_base
from sqlalchemy import  create_engine 
from sqlalchemy.orm import sessionmaker


engine = create_engine('postgresql+psycopg2://asaini:root@localhost/pypolls', echo = True)

Base = declarative_base()

SessionLocal = sessionmaker(bind = engine)


# %%

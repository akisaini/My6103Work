#%%
from db import Base
from models import Item
from db import engine


Base.metadata.create_all(engine)


# %%

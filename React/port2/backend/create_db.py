#%%
from db import Base
from models import VideoModel
from db import engine


Base.metadata.create_all(engine)

# %%

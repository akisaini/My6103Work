#%%
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
#%%
img = cv2.imread('Sports player classification/test_images/sharapova1.jpg')
# %%
# using matplotlib to view the image. 
plt.imshow(img)
# %%
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_img.shape
# %%
plt.imshow(gray_img, cmap = 'gray')

# %%
face_cascade = cv2.CascadeClassifier('./opencv/frontface.xml')
eyes_cascade = cv2.CascadeClassifier('./opencv/eyes.xml')
faces = face_cascade.
# %%

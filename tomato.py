#%% 
import tensorflow as tf
from tensorflow.keras import models, layers 
import matplotlib.pyplot as plt
import numpy as np
#%%
batch_size = 32
image_size = 256
ds = tf.keras.utils.image_dataset_from_directory(
    './tomato_cnn',
    shuffle = True,
    batch_size = batch_size,
    image_size = (image_size,image_size),  
)

# %%
# Splitting data into train/test/validation
len(ds) # length is 141 because batch size is 32. 
# %%
train_ds = len(ds)*0.75
train_ds # 105
# %%
train_ds = ds.take(105)
# %%
test_ds = ds.skip(105)
# %%
val_ds = ds.take(int(len(ds)*.15))
len(val_ds) # 21
# %%
test_ds = test_ds.skip(len(val_ds))
len(test_ds) # 15
# %%
# class_names

class_names = ds.class_names

# %%
# Prefetch and cache set up:
train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size = tf.data.AUTOTUNE)
#%%
# Setting up 'Resizing and rescaling' and 'Data Augumentation' preprocessing layers: 
resize_and_rescale = tf.keras.Sequential([
  layers.Resizing(image_size, image_size),
  layers.Rescaling(1./255)
])

data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])
#%%
model = models.Sequential([
  # Adding the preprocessing layers created earlier.
  resize_and_rescale,
  data_augmentation,
  layers.Conv2D(25, (3,3), padding='same', activation='relu', input_shape = (batch_size, image_size, image_size, 3)),
  layers.MaxPooling2D(),
  layers.Conv2D(25, (3,3), activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(25, (3,3), activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(25, (3,3), activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(25, (3,3), activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(96, activation = 'relu'),
  layers.Dense(len(class_names), activation = 'softmax')
])
# %%
model.build(input_shape = (batch_size, image_size, image_size, 3))
# 3 is RGB
# %%
model.summary()
# %%
model.compile(optimizer = 'adam',
              loss = tf.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics = ['accuracy']
              )
# %%
history = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = 10
)
# %%
accuracy = history.history['accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']
# %%
plt.figure(figsize = (10,10))
plt.plot(range(10), accuracy, label = 'Accuracy')
plt.plot(range(10), loss, label = 'Loss')
plt.title('Accuracy vs Loss')
plt.legend()
plt.show()
#%%
score = model.evaluate(test_ds)
# test_ds has a near accuracy of 92%, which is similar to train_ds. 
# %%
# takes in model and image( a tf image) as input. - No need to covert to numpy here as the tf preprocessing converts the image into and array. 
def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100*(np.max(predictions[0])), 2)
    return predicted_class, confidence
#%%

plt.figure(figsize = (30, 30))
for image_batch, label_batch in test_ds.take(1):
    for i in range(18):
        ax = plt.subplot(6,3,i+1)
        predicted_class, confidence = predict(model, image_batch[i])
        plt.imshow(image_batch[i].numpy().astype('uint8'))
        plt.axis('off')
        plt.title(f'Actual Class:{class_names[label_batch[i].numpy()]}\nPredicted Class:{predicted_class}\n Confidence: {confidence}')
        
# %%
model.save('./tomato_cnn/')
# %%

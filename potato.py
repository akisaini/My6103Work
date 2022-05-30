#%%
import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
#%%
IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3 # rgb
dataset = tf.keras.preprocessing.image_dataset_from_directory(
     "potato_cnn",
     shuffle = True,
     image_size = (IMAGE_SIZE, IMAGE_SIZE),
     batch_size = BATCH_SIZE
 )
# %%
class_names = dataset.class_names
class_names
# %%
len(dataset) # 68
# This is because the dataset is divided into batches of 32. 
# The total length of the dataset is 68. 68*32  = total size of data set
#%%
for image_batch, label_batch in dataset.take(1):
    plt.imshow(image_batch[0].numpy().astype('uint8'))
    plt.title(class_names[label_batch[0].numpy()])
    plt.axis('off')
# %%
train_size = 0.8
len(dataset)*(train_size) 
train_ds =dataset.take(54)
# %%
# 20% total, split into 10% validation and 10% actual test set
test_size = 0.2
test_ds = dataset.skip(54)
# %%
val_size = 0.1
len(dataset)*val_size
#%%
val_ds = test_ds.take(6)
test_ds = test_ds.skip(6)
#%%
# Setting up cache and prefetch
# Prefetch - CPU loads the upcoming batch while the GPU is processing the first one. Generally more useful in case of a CPU/GPU set up. 

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
 
#%%
# Resizing and rescaling using tf preprocessing:
# All images resized to 256*256
resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE,IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1.0/255)
])
#%%
# Data Augumentation using preprocessing:
# This helps to generate more data using the existing images. 
# We use the existing images and add more fiters to it. For example: Rotate left, zoom, contrast, horizontal rotation. 
# this effectively helps to create more data using the existing data.
data_augumentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2)
])
#
# Above we have created objects for resize_and_rescale and data_augumentation which will be used when building the 'Sequential' tf model. 
# %%
input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 3

model = models.Sequential([
    resize_and_rescale,
    data_augumentation,
    layers.Conv2D(32, (3,3), activation = 'relu', input_shape = input_shape),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32, (3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32, (3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32, (3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32, (3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation = 'relu'),
    layers.Dense(n_classes, activation = 'softmax')
])
# %%
model.build(input_shape = input_shape)
# %%
model.summary()
# %%
model.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
    metrics = ['accuracy']
)
# %%
history = model.fit(train_ds,
          epochs = 20,
          batch_size = BATCH_SIZE,
          verbose = 1,
          validation_data = val_ds)
# %%
scores = model.evaluate(test_ds)
# %%
history.params
# %%
history.history.keys()
# %%
accuracy = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']
# %%
# x = independent variable
# y = dependent variable
plt.plot(range(history.params['epochs']), accuracy, label = 'Training Accuracy')
plt.plot(range(history.params['epochs']), loss, label = 'Loss Value')
plt.legend()
plt.title('Training vs Loss')
plt.show()
# %%
import numpy as np
# Take one batch. One batch is size of 32. 
for image_batch, label_batch in test_ds.take(1):
    img = image_batch[0].numpy().astype('uint8')
    label = label_batch[0].numpy()

    print('Img to predict:')
    plt.imshow(img)
    print('Actual Label:', class_names[label])
    
    batch_pred = model.predict(image_batch)
    # batch_pred has predicted values for all 32 images in the batch
    print('Predicted Label:', class_names[np.argmax(batch_pred[0])])
    # Since we chose the number of nuerons in the last Dense layer as n_classes which equals 3, we have three predicted values. 
    # We choose the highest value using np.argmax() and feed that in class_names[].
    
# %%

def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf. expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100*(np.max(predictions[0])), 2)
    return predicted_class, confidence
#%%
plt.figure(figsize=(15,15))
for images, labels in test_ds.take(1):
    for i in range(9):    
        ax = plt.subplot(3,3,i+1)
        # (i+1) is the position of each fig/plot
        plt.imshow(images[i].numpy().astype('uint8'))
        predicted_class, confidence = predict(model, images[i].numpy())
        plt.axis('off')
        plt.title(f'Actual Class:{class_names[labels[i].numpy()]} \nPredicted Class:{predicted_class}\n Confidence:{confidence}')
# %%
model.save('./potato_cnn/1')
# %%

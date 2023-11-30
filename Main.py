# Description of the Brain tumor Dataset
#The dataset is organized into 2 folders (Training, Testing) and contains 4 subfolders for each image category. There are 3,264 MRI images (JPEG) and 4 categories (Glioma/Meningioma/Pituitary/No_tumor).

#Training set
#Glioma tumor (826 images)
Meningioma tumor (822 images)
#No tumor (395 images)
#Pituitary tumor (827 images)
#Testing set
#Glioma tumor (100 images)
#Meningioma tumor (115 images)
#No tumor (105 images)
#Pituitary tumor (74 images)
#Importing the necessary libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv
# import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
from IPython.display import display, clear_output
import ipywidgets as widgets
import io
import os
# import cv2
import requests
from tqdm import tqdm
import numpy as np
from io import BytesIO
from PIL import Image
from keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Activation,Concatenate, BatchNormalization


labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
X_train = []
y_train = []
X_test = []
y_test = []
image_size = 224

# Function to download and resize images from a URL
def download_and_resize_images(url, label):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    img = cv.resize(img, (image_size, image_size))
    # img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # img = cv2.resize(img, (image_size, image_size))
    return img, label

# Training data
for label in labels:
    folder_url = f'https://drive.google.com/drive/folders/1h2jisdtZGNWohMkwwZRnSvm-5s7ClopY/'
    for filename in tqdm(os.listdir(folder_url)):
        img, label = download_and_resize_images(os.path.join(folder_url, filename), label)
        X_train.append(img)
        y_train.append(label)

# Testing data
for label in labels:
    folder_url = f'https://drive.google.com/drive/folders/1MYzwCTiinE-5QmbmzUVt5rv5IQ50fXUE/'
    for filename in tqdm(os.listdir(folder_url)):
        img, label = download_and_resize_images(os.path.join(folder_url, filename), label)
        X_test.append(img)
        y_test.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)

# Previewing the images in each classes
k=0
fig, ax = plt.subplots(1,4,figsize=(20,20))
fig.text(s='Sample Image From Each Label',size=18,fontweight='bold',
             fontname='monospace',y=0.62,x=0.4,alpha=0.8)
for i in labels:
    j=0
    while True :
        if y_train[j]==i:
            ax[k].imshow(X_train[j])
            ax[k].set_title(y_train[j])
            ax[k].axis('off')
            k+=1
            break
        j+=1

        # Shuffle the train set
        X_train, y_train = shuffle(X_train,y_train, random_state=14)

        # Print out train set shape
        X_train.shape

        # Print out test set shape
        X_test.shape

        # Show the counts of observations in each categorical
        sns.countplot(y_train)

        sns.countplot(y_test)


        # Performing One Hot Encoding on the labels after converting it into numerical values

        y_train_new = []
        for i in y_train:
            y_train_new.append(labels.index(i))
            y_train = y_train_new
            y_train = tf.keras.utils.to_categorical(y_train)
            
            y_test_new = []
            for i in y_test:
                y_test_new.append(labels.index(i))
                y_test = y_test_new
                y_test = tf.keras.utils.to_categorical(y_test)

# Dividing the dataset into Training and Testing sets.
            
        X_train,X_val,y_train,y_val = train_test_split(X_train,y_train, test_size=0.1,random_state=14)

# Normalize the data


        X_train = np.array(X_train) / 255.
        X_val = np.array(X_val) / 255.
        X_test = np.array(X_test) / 255.

        #  Data Augmentation To prevent the problem of overfitting, we can artificially enlarge the dataset. 
        # I can increase the size of the current dataset. 
        # The idea is to alter the training data with small transformations to reproduce the variations. 
        # Data augmentation strategies are methods for changing the array representation while keeping the label the same while altering the training data. 
        # Grayscales, horizontal and vertical flips, random crops, color jitters, translations, rotations, and many other augmentations are popular. 
        # I can easily double or increase the number of training examples by applying only a couple of these changes to the training data.

        datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.1,  # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
        
        datagen.fit(X_train)

#         For the data augmentation, i choosed to :

# Randomly rotate some training images by 10 degrees.
# Randomly Zoom by 10% some training images.
# Randomly shift images horizontally by 10% of the width.
# Randomly shift images vertically by 10% of the height.
# Randomly flip images horizontally.
# Callback function

# Callbacks can help fix bugs more quickly, and can help build better models. They can help you visualize how your model’s training is going, and can even help prevent overfitting by implementing early stopping or customizing the learning rate on each iteration.

# By definition, "A callback is a set of functions to be applied at given stages of the training procedure. You can use callbacks to get a view on internal states and statistics of the model during training."

# In this notebook, I'll be using TensorBoard, ModelCheckpoint and ReduceLROnPlateau callback functions

        tensorboard = TensorBoard(log_dir = 'logs')
        checkpoint = ModelCheckpoint("effnet.h5",monitor="val_accuracy",save_best_only=True,mode="auto",verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.3, patience = 2, min_delta = 0.001, mode='auto',verbose=1)

#  Detail of model implementation
        
        model_cnn = Sequential()
        
        model_cnn.add(Conv2D(64, (3, 3), padding='same',input_shape=(image_size,image_size,3))) 
        model_cnn.add(Activation('relu'))
        model_cnn.add(BatchNormalization())
        model_cnn.add(Conv2D(64, (3, 3))) 
        model_cnn.add(Activation('relu'))
        model_cnn.add(MaxPooling2D(pool_size=(2, 2))) 
        model_cnn.add(BatchNormalization())
        model_cnn.add(Dropout(0.35))
        
        model_cnn.add(Conv2D(64, (3, 3), padding='same'))
        model_cnn.add(Activation('relu'))
        model_cnn.add(BatchNormalization()) 
        
        model_cnn.add(Conv2D(64, (3, 3)))
        model_cnn.add(Activation('relu'))
        model_cnn.add(MaxPooling2D(pool_size=(2, 2))) 
        model_cnn.add(BatchNormalization())
        model_cnn.add(Dropout(0.35)) 
        
        model_cnn.add(Conv2D(64, (3, 3), padding='same')) 
        model_cnn.add(Activation('relu'))
        model_cnn.add(BatchNormalization())
        
        model_cnn.add(Flatten()) 
        model_cnn.add(Dropout(0.5)) 
        model_cnn.add(Dense(512)) 
        model_cnn.add(Activation('relu'))
        model_cnn.add(BatchNormalization())
        model_cnn.add(Dense(4)) 
        model_cnn.add(Activation('softmax'))
        
        model_cnn.summary()

        from keras.utils.vis_utils import plot_model
        plot_model(model_cnn, to_file='model_cnn_plot.png', show_shapes=True, show_layer_names=True)

        model_cnn.compile(optimizer = 'adam',
               loss = 'categorical_crossentropy',
               metrics = ['accuracy'])

        history = model_cnn.fit(X_train, y_train,validation_split=0.1, verbose=1, batch_size = 32, validation_data = (X_val, y_val),
                     epochs = 20, callbacks =[tensorboard,checkpoint,reduce_lr])

        model_cnn.save('cnn_model.h5') 
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        
        epochs = range(1, len(acc) + 1)
        
        plt.plot(epochs, acc, 'r', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        
        plt.figure()
        
        plt.plot(epochs, loss, 'r', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()

        








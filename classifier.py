
# Utilities
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import copy

# Tensorflow utilitiues
import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers.experimental import preprocessing

# For interactive timer
import tqdm as tqdm


import sklearn
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.metrics import classification_report





# The dataset used is available at https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
# The data needs to be downloaded and copied to the following folders

train_dir = '/kaggle-covid19-data/input/chest-xray-pneumonia/chest_xray/train'
test_dir = '/kaggle-covid19-data/chest-xray-pneumonia/chest_xray/test'
val_dir = '/kaggle-covid19-data/input/chest-xray-pneumonia/chest_xray/val'


# Function to show some of the scans from a given dataset, with labels
def plot_scans_from_data(ds):
    plt.figure(figsize=(8,8))

    for img,labels in train_ds.take(3):
        for scan in range(9):
            ax = plt.subplot(3,3,scan+1)
            plt.imshow(np.squeeze(img[scan].numpy().astype(np.uint8)))
            plt.title(train_ds.class_names[labels[scan]])
            plt.show()


# Image dimensions
HEIGHT = 200
WIDTH = 200

# Preprocessing train, test and val data
BATCH_SIZE = 32

train_ds = keras.preprocessing.image_dataset_from_directory(train_dir, color_mode='grayscale',image_size=(WIDTH, HEIGHT),batch_size=BATCH_SIZE)
test_ds = keras.preprocessing.image_dataset_from_directory(test_dir,color_mode='grayscale',image_size=(WIDTH, HEIGHT),batch_size=BATCH_SIZE)
val_ds = keras.preprocessing.image_dataset_from_directory(val_dir,color_mode='grayscale',image_size=(WIDTH, HEIGHT),batch_size=BATCH_SIZE)


# Plotting some of the figures from the dataset with the labels


plot_scans_from_data(train_ds)
plot_scans_from_data(test_ds)

# Performing data augmentation -- this seems to quite popular in medical datasets
# The parameters for flips, rotation and zoom are taken empirically
aug_ds = tf.keras.Sequential([
    preprocessing.RandomFlip('horizontal'),
    preprocessing.RandomRotation(0.1),
    preprocessing.RandomZoom(0.1),
])

# Optimization code for CPU
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.map(lambda x,y: (aug_ds(x),y)).cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


# Model architecture

model = tf.keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.2),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam', loss = tf.losses.BinaryCrossentropy(), metrics = ['accuracy'])


EPOCHS = 20
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)


# Evaluating the model on the test dataset
model.evaluate(test_ds)

# Observed results
# Test Accuracy: 80%

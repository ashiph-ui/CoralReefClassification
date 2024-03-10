import numpy as np
import pandas as pd
import os
import cv2
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout



class CNN:
    def  __init__(self, bleached_corals, healthy_corals):
        self.bleached_corals = bleached_corals
        self.healthy_corals = healthy_corals
    
    def load_images(self):
        images = []
        labels = []
        for img in os.listdir(self.bleached_corals):
            img = cv2.imread(os.path.join(self.bleached_corals, img))
            img = cv2.resize(img, (240, 300))
            images.append(img)
            labels.append(1)

        for img in os.listdir(self.healthy_corals):
            img = cv2.imread(os.path.join(self.healthy_corals, img))
            img = cv2.resize(img, (240, 300))
            images.append(img)
            labels.append(0)
        return images, labels


    def shuffle_images(self, images, labels):
        images, labels = self.load_images()
        images = np.array(images)
        labels = np.array(labels)
        shuffle = list(zip(images, labels)) 
        random.shuffle(shuffle)
        images, labels = zip(*shuffle)
        return images, labels
    
 
    def preprocess_images(self, images, labels):
        images, labels = self.shuffle_images(images, labels)
        images = np.array(images)
        images = images / 255
        images = [cv2.resize(img, (240, 300)) for img in images]
        return images, labels
    
    def DataAugmentation(self, images, labels):
        images, labels = self.preprocess_images(images, labels)
        augmented_images = []
        augmented_labels = []
        for i in range(len(images)):
            augmented_images.append(images[i])
            augmented_labels.append(labels[i])
            augmented_images.append(cv2.flip(images[i], 1))
            augmented_labels.append(labels[i])
            augmented_images.append(cv2.flip(images[i], 0))
            augmented_labels.append(labels[i])
            
        return augmented_images, augmented_labels
    
   

    def train_test_split(self, images, labels):
        images, labels = self.DataAugmentation(images, labels)
        X_train = np.array(images[:int(len(images) * 0.8)])  # Convert to NumPy array
        Y_train = np.array(labels[:int(len(labels) * 0.8)])   # Convert to NumPy array
        X_test = np.array(images[int(len(images) * 0.8):])    # Convert to NumPy array
        Y_test = np.array(labels[int(len(labels) * 0.8):])     # Convert to NumPy array
        return X_train, Y_train, X_test, Y_test
    

    def create_model(self):
        model = Sequential()
        
    
        model.add(Conv2D(12, (3, 3), input_shape=(300, 240, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        # model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.1))
        
  

        model.add(Flatten())
        model.add(Dense(units=32, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
        
    def train_model(self):
        images, labels = self.load_images()
        X_train, Y_train, X_test, Y_test = self.train_test_split(images, labels)
        model = self.create_model()
        print(X_train.shape[0])
        model.fit(X_train, Y_train, epochs=50, batch_size=64, validation_data=(X_test, Y_test))
        return model
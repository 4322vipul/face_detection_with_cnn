#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 17:05:26 2018

@author: vipul
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
import pandas as pd
import cv2
import glob
import os

#CNN Model

classifier=Sequential()

classifier.add(Convolution2D(32,3,3,input_shape=(120,160,3),activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

classifier.add(Dense(output_dim=128,activation='relu'))
classifier.add(Dense(output_dim=1,activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


#Fitting model to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)

training_set=train_datagen.flow_from_directory('/home/vipul/Desktop/cam_face_recog/pics',
                                               target_size=(120,160),
                                               batch_size=32,
                                              class_mode='binary')
vipul_set=test_datagen.flow_from_directory('/home/vipul/Desktop/cam_face_recog/pics1',
                                           target_size=(120,160),
                                           batch_size=1,
                                           class_mode='binary')

classifier.fit_generator(training_set,
                         samples_per_epoch=100,
                         nb_epoch=25, validation_data=vipul_set,nb_val_samples=2)


model_json=classifier.to_json()
with open ("model1.json","w") as json_file:
    json_file.write(model_json)
    
classifier.save_weights("model1.h5")
print("saved model to disk")                    
                         

sample_img=image.load_img('/home/vipul/Desktop/cam_face_recog/pics1/test_set/v.jpg',target_size=(120,160))
sample_img=image.img_to_array(sample_img)
sample_img=np.expand_dims(sample_img,axis=0)

result=classifier.predict(sample_img)
training_set.class_indices
if result[0][0]==1:
    prediction='Match Found'
else:
    prediction='Match not found'

print('Result',prediction)   

'''
test=vipul_set[0]
test1=test[0]
test2=test1.reshape([64,64,3])model 
plt.imshow(test2)
plt.show()

dataset=[]

for i in range(0,20):
    train=training_set[0]
    train1=train[0]
    train2=np.array(train1[i])
    dataset.insert(i,train2)
    plt.imshow(train2)
    plt.show()
'''

'''
first=training_set[0]
fir=first[0]
fir.shape
f=fir[5]
plt.imshow(f)
plt.show()
'''

'''
images_train=[]
for filename in os.listdir('/home/vipul/Desktop/cam_face_recog/pics/training_set'):
    img=cv2.imread(os.path.join('/home/vipul/Desktop/cam_face_recog/pics/training_set',filename))
    if img is not None:
            images_train.append(img)
            
images_test=[]
img1=cv2.imread('/home/vipul/Desktop/cam_face_recog/pics1/test_set/v.jpg')  
images_test.append(img1)     
b=images_test[:,2]     
'''
# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import load_model
from keras.callbacks import TensorBoard
from time import time
import keras.backend as K

config = K.tf.ConfigProto()
config.gpu_options.allow_growth = True
session = K.tf.Session(config=config)

#Initialising CNN
network = Sequential()

# Convolution 1
network.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Pooling1
network.add(MaxPooling2D(pool_size = (2, 2)))

# Convolution 2
network.add(Conv2D(32, (3, 3), activation = 'relu'))

#Pooling 2
network.add(MaxPooling2D(pool_size = (2, 2)))

#Flattening
network.add(Flatten())

#Fully connected layer 1
network.add(Dense(units = 160, activation = 'relu'))

#Fully connected layer 2
network.add(Dense(units = 128, activation = 'relu'))

#Output layer
network.add(Dense(units = 1, activation = 'sigmoid'))

#Compiling the CNN
network.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

#Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
network.fit_generator(training_set,
                         steps_per_epoch = 4000,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 2000,
                         callbacks=[tensorboard])

#Saving model
path = ('cats.h5')
network.save(path)
del network
network = load_model(path)


#SinglePrediction

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = network.predict(test_image)
training_set.class_indices
if result[0][0] == 0:
    prediction = 'cat'
else:
    prediction = 'unknown object'
    
predicted_classes = network.decode_predictions(prediction, top=1)
imagenet_id, name, confidence = predicted_classes[0][0]
print("This is a {} with {:.4}% confidence!".format(name, confidence * 100))
    
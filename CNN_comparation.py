# -*- coding: utf-8 -*-
from keras.models import load_model

network = load_model('dogs.h5')

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = network.predict(test_image)
training_set.class_indices
if result[0][1] == 0:
    prediction = 'cat'
else:
    prediction = 'unknown object'
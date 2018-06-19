# -*- coding: utf-8 -*-
from keras.models import load_model
import numpy as np
from keras.preprocessing import image

dogs = load_model('dogs.h5')
cats = load_model('cats.h5')
single = load_model('dogs_and_cats.h5')


test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

result = cats.predict(test_image)
result2 = dogs.predict(test_image)
if result2[0][0] == 1 and result[0][0] == 0:
    prediction_both = 'cat'
elif result2[0][0] == 0 and result[0][0] == 1:
    prediction_both = 'dog'
else:
    prediction_both ='unknown object'
    
result3 = single.predict(test_image)
if result3[0][1] == 0:
    prediction_single = 'cat'
else:
    prediction_single = 'dog'
# -*- coding: utf-8 -*-
import numpy as np
from keras.preprocessing import image


def image_to_tensor(filename, img_height, img_width):
    img = image.load_img(filename, target_size=(img_height, img_width))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # normalization
    x = x / 255.0
    return x

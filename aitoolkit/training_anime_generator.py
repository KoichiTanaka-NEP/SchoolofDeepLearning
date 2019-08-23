# -*- coding: utf-8 -*-

import numpy as np
from keras.callbacks import Callback
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
# from PIL import Image
from utils import image_to_tensor
from IPython.display import Image, display


class TrainingAnimeGenerator(Callback):
    def __init__(self, img_height, img_width, labels, test_img_path, result_gif_path, color_mode='rgb'):
        self.test_img_path = test_img_path
        self.result_gif_path = result_gif_path
        self.img_height = img_height
        self.img_width = img_width
        self.labels = labels
        self.now_epoch = 1
        self.color_mode = color_mode
        self.gif_images = []
        sns.set(style="whitegrid")


    def on_train_begin(self, logs={}):
        self.verbose = self.params['verbose']
        self.fig = plt.figure()


    def on_epoch_begin(self, epoch, logs={}):
        print('------------------------------------------------------------------------------------------------')
        print('------------------------------------------------------------------------------------------------')


    def on_epoch_end(self, epoch, logs={}):
        if self.color_mode == "rgb":
            input_tensor = image_to_tensor(self.test_img_path, self.img_height, self.img_width)
        elif self.color_mode == "grayscale":
            input_tensor = image_to_tensor(self.test_img_path, self.img_height, self.img_width, color_mode="grayscale")
        detection = self.model.predict(input_tensor)[0]
        a = np.array(detection)
        sns.set(style="white", context="talk")
        f, ax1 = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
        gif_img = sns.barplot(self.labels, detection, palette="PiYG", ax=ax1)
        self.gif_images.append([gif_img])


    def on_train_end(self, logs={}):
        ani = animation.ArtistAnimation(self.fig, self.gif_images, interval=100)
        ani.save(self.result_gif_path+'training.gif', writer="imagemagick")

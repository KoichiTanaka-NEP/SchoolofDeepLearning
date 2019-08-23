# -*- coding: utf-8 -*-

import numpy as np
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.layers import Convolution2D, MaxPooling2D
from keras import optimizers
from keras.callbacks import TensorBoard
from keras.callbacks import ReduceLROnPlateau
from graph_generator import GraphGenerator
from training_anime_generator import TrainingAnimeGenerator
from utils import image_to_tensor
import os
import glob


class ObjectDetector:
    def __init__(self, num_of_detection = 1000, train_data_dir = 'data/train', validation_data_dir = 'data/validation', result_data_dir = 'results/object_detection'):
        self.is_created_datasets = False
        self.is_created_model = False
        self.is_trained = False
        self.img_width = 150
        self.img_height = 150
        self.train_data_dir = train_data_dir
        dir_labels = glob.glob(self.train_data_dir+'/*')
        self.num_of_detection = len(dir_labels)
        self.validation_data_dir = validation_data_dir
        self.result_data_dir = result_data_dir
        self.model = None


    def create_datasets(self, labels=None, color_mode='rgb', batch_size=32, is_whitening = False):
        train_datagen = ImageDataGenerator(rescale=1.0 / 255, width_shift_range=0.25, height_shift_range=0.25, \
                                           rotation_range=45, shear_range=0.2, zoom_range=0.2, channel_shift_range=20, \
                                           horizontal_flip=True, vertical_flip=True, zca_whitening=is_whitening)
        validation_datagen = ImageDataGenerator(rescale=1.0 / 255, zca_whitening=is_whitening)
        if is_whitening is True:
            ## Warning : This function has not been implemented yet!!
            train_datagen.fit()
            validation_datagen.fit()
        else:
            pass
        self.train_generator = train_datagen.flow_from_directory(self.train_data_dir, \
            target_size=(self.img_height, self.img_width), batch_size=batch_size, \
            class_mode='categorical', classes=labels, shuffle=True, color_mode=color_mode)
        self.validation_generator = validation_datagen.flow_from_directory(self.validation_data_dir,\
            target_size=(self.img_height, self.img_width), batch_size=batch_size, \
            class_mode='categorical', classes=labels, shuffle=True, color_mode=color_mode)
        self.is_created_datasets = True
        if labels is not None:
            self.labels = labels
        else:
            dir_labels = glob.glob(self.train_data_dir+'/*')
            labels = []
            for dir_label in dir_labels:
                strs = dir_label.split('/')
                # print strs[len(strs)-1]
                labels.append(strs[len(strs)-1])
            self.labels = labels
        self.color_mode = color_mode


    def create_model(self, sgd_lr=1e-4, sgd_momentum=0.9, original_model=None, channel_num = 3):
        input_tensor = Input(shape=(self.img_height, self.img_width, channel_num))
        if original_model is not None:
            self.input_model = original_model
        else:
            self.input_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
        # Fine-tuning Model
        tuning_model = Sequential()
        tuning_model.add(Flatten(input_shape=self.input_model.output_shape[1:]))
        tuning_model.add(Dense(256, activation='relu'))
        tuning_model.add(Dropout(0.5))
        tuning_model.add(Dense(self.num_of_detection, activation='softmax'))
        self.model = Model(input=self.input_model.input, output=tuning_model(self.input_model.output))
        # for layer in self.model.layers[:15]:
            # layer.trainable = False
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=sgd_lr, momentum=sgd_momentum), metrics=['accuracy'])
        self.is_created_model = True


    def get_model_summary(self):
        if self.is_created_model is None or self.is_created_model is False:
            print("モデルをまだ作れていないよ。")
        else:
            self.model.summary()


    def fit(self, nb_train_samples = 1000, nb_validation_samples = 800, nb_epoch = 20, test_image_dir = './detect_data/', log_dir='./tb_logs'):
        if self.is_created_datasets is None or self.is_created_datasets is False:
            print("データセットを先に作ってね。")
        else:
            if self.is_created_model is None or self.is_created_model is False:
                print("モデルを先に作ってね。")
            else:
                query = test_image_dir + '*.jpg'
                self.test_img_list = glob.glob(query)
                # Callbacks
                graph_generator = GraphGenerator(img_height = self.img_height, img_width = self.img_width,\
                                                 labels = self.labels, test_img_list=self.test_img_list, color_mode=self.color_mode)
                training_anime_generator = TrainingAnimeGenerator(img_height = self.img_height, img_width = self.img_width,\
                                                 labels = self.labels, test_img_path=self.test_img_list[0], result_gif_path='./results/', color_mode=self.color_mode)
                tensorboard = TensorBoard(log_dir=log_dir)
                reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
                self.history = self.model.fit_generator(self.train_generator,samples_per_epoch=nb_train_samples, \
                                                        nb_epoch=nb_epoch, validation_data=self.validation_generator, \
                                                        nb_val_samples=nb_validation_samples,
                                                        callbacks=[graph_generator,tensorboard,reduce_lr])


    def detect(self, filename):
        input_tensor = image_to_tensor(filename, self.img_height, self.img_width)
        detection = self.model.predict(input_tensor)[0]
        a = np.array(detection)
        detect_label = self.labels[a.argmax(0)]
        print detect_label
        print detection


    def output_history(self, result_file):
        result = os.path.join(self.result_data_dir, result_file)
        history = self.history
        loss = history.history['loss']
        acc = history.history['acc']
        val_loss = history.history['val_loss']
        val_acc = history.history['val_acc']
        nb_epoch = len(acc)
        with open(result, "w") as f:
            f.write("epoch\tloss\tacc\tval_loss\tval_acc\n")
            for i in range(nb_epoch):
                f.write("%d\t%f\t%f\t%f\t%f\n" % (i, loss[i], acc[i], val_loss[i], val_acc[i]))


    def deactivate_layer(self, layer_num):
        layer = self.model.layers[layer_num]
        layer.trainable = False


    def dump_model_weights(self, model_path, weights_path):
        json_string = self.model.to_json()
        open(model_path, 'w').write(json_string)
        self.model.save_weights(weights_path)


    def save_model(self):
        self.dump_model_weights("/home/ec2-user/nes_application/src/model/model.json", "/home/ec2-user/nes_application/src/model/weights.hdf5")
        self.dump_label("/home/ec2-user/nes_application/src/model/labels.txt")


    def dump_label(self, txt_path):
        with open(txt_path, "w") as f:
            output_string = ""
            for label in self.labels:
                output_string = output_string + label
                output_string = output_string + " "
            f.write(output_string)

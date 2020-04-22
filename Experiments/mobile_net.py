import numpy as np
from keras.preprocessing import image
import keras
from keras import backend as K
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
# %matplotlib inline

from mpl_toolkits.mplot3d import Axes3D
from sklearn import model_selection
import sklearn
from pathlib import Path
from skimage import io
import handshape_datasets as hd
import os
from os import path
from experiment import Experiment
import logging

class MobileNet(Experiment):

    def __init__(self, epochs, batch_size, dataset_id, **kwargs):
        super().__init__("MobileNet", dataset_id, epochs, batch_size)
        if 'version' in kwargs:
            ver=kwargs['version']
        if 'delete' in kwargs:
            supr= kwargs['delete']
        try:
            self.dataset = hd.load("lsa16", version=ver, delete=supr)
        except:
            try:
                self.dataset=hd.load("lsa16", version=ver)
            except:
                try:
                    self.dataset=hd.load("lsa16", delete=supr)
                except:
                    self.dataset = hd.load("lsa16")
        self.model_name = "MobileNet"
        self.input_shape = self.dataset[0][0].shape
        self.classes = self.dataset[1]['y'].max() + 1
        self.history = ""

    def get_loader(self)->Experiment:
        return MobileNet()

    def get_history(self):
        return self.history

    def load(self, model, X_train, Y_train, X_test, Y_test):
        self.history = model.fit(X_train, Y_train, batch_size=self.batch_size, epochs=self.epochs,
                                 validation_data=(X_test, Y_test))

        return self.history

    def get_result(self):
        path = self.get_path()

        loss_history = self.history.history["loss"]
        acc_history = self.history.history["accuracy"]
        val_loss_history = self.history.history["val_loss"]
        val_acc_history = self.history.history["val_accuracy"]

        numpy_loss_history = np.array(loss_history)
        np.savetxt(os.path.join(path,"loss_history.txt"), numpy_loss_history, delimiter=",", fmt='%0.2f')

        numpy_acc_history = np.array(acc_history)
        np.savetxt(os.path.join(path,"acc_history.txt"), numpy_acc_history, delimiter=",",fmt='%0.2f')

        numpy_val_loss_history = np.array(val_loss_history)
        np.savetxt(os.path.join(path,"val_loss_history.txt"), numpy_val_loss_history, delimiter=",",fmt='%0.2f')

        numpy_val_acc_history = np.array(val_acc_history)
        np.savetxt(os.path.join(path,"val_acc_history.txt"), numpy_val_acc_history, delimiter=",",fmt='%0.2f')

        return True

    def split(self, test_size):
        X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(self.dataset[0], self.dataset[1]['y'],
                                                                                    test_size=test_size,
                                                                                    stratify=self.dataset[1]['y'])
        return X_train, X_test, Y_train, Y_test

    def build_model(self):
        base_model = keras.applications.mobilenet.MobileNet(input_shape=self.input_shape, weights='imagenet',
                                                            include_top=False)
        output = keras.layers.GlobalAveragePooling2D()(base_model.output)
        output = keras.layers.Dense(32, activation='relu')(output)
        # Nueva capa de salida
        output = keras.layers.Dense(self.classes, activation='softmax')(output)  # cambiar cantidad de clases
        model = Model(inputs=base_model.input, outputs=output)
        # Entrenar con nuevos datos
        model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        #y_true = model.fit_generator()
        #y_pred = model.predict_generator(model.fit_generator)

        return model

    # lsa con batck size 32

    def graphics(self):
        path = self.get_path()
        graphic_acc_file=os.path.join(path,"figure_acc.png")
        graphic_loss_file = os.path.join(path, "figure_loss.png")
        self.plot_training_curves(self.history,graphic_acc_file, graphic_loss_file)
        #self.plot_confusion_matrix(y_true, y_pred)

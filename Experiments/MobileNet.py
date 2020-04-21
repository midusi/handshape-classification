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
#%matplotlib inline

from mpl_toolkits.mplot3d import Axes3D
from sklearn import model_selection
import sklearn
from pathlib import Path
from skimage import io
import handshape_datasets as hd
import os
from os import path
import Experiment
from ..Experiment import Experiment
import logging

class MobileNet(Experiment):

    def __init__(self, epochs, batch_size, dataset_id):
        super().__init__("MobileNet", dataset_id, epochs, batch_size)
        self.dataset = hd.load(dataset_id)
        self.model_name="MobileNet"
        self.input_shape = self.dataset[0][0].shape
        self.classes= self.dataset[1]['y'].max()+1
        self.history=""


    def get_loader(self)->Experiment:
        return MobileNet()

    def get_history(self):
        return self.history

    def load(self, model,X_train, Y_train, X_test, Y_test, batch_size, epochs):
        self.history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, Y_test))
        return self.history

    def get_result(self):
        path=self.get_path()
        log_path=os.path.join(path, 'history.log')
        logging.basicConfig(filename=log_path, level=logging.DEBUG)
        logging.info(self.get_history())
        return True

    def split(self,dataset, test_size):
        X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(dataset[0], dataset[1]['y'],                                                                           test_size=test_size,                                                                           stratify=dataset[1]['y'])
        return X_train, X_test, Y_train, Y_test

    def build_model(self):

        base_model = keras.applications.mobilenet.MobileNet(input_shape=self.input_shape,weights='imagenet',include_top=False)
        output = keras.layers.GlobalAveragePooling2D()(base_model.output)
        output = keras.layers.Dense(32,activation='relu')(output)
        # Nueva capa de salida
        output = keras.layers.Dense(self.classes,activation='softmax')(output) #cambiar cantidad de clases
        model=Model(inputs=base_model.input,outputs=output)
        #Entrenar con nuevos datos
        model.compile(optimizer='Adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

        return True
    #lsa con batck size 32

    def graphics(self,history, y_true, y_pred):
        self.plot_training_curves(history)
        self.plot_confusion_matrix(y_true, y_pred)
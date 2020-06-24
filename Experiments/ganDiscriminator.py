from os import listdir

import numpy as np
import keras
from keras.models import Model
import sklearn
from skimage import transform
import tensorflow as tf
from keras.models import load_model
import handshape_datasets as hd
from PIL import Image
import os
from experiment import Experiment
from prettytable import PrettyTable
from sklearn import model_selection
from pathlib import Path
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense

gan_folder = Path.home() / 'handshape-classification' / 'GANResults'

class ganDiscriminator(Experiment):

    def __init__(self, epochs, batch_size, dataset_id, **kwargs):
        self.tl = True

        load_path = os.path.join(gan_folder, dataset_id)
        files = list(
            filter(lambda x: ".h5" in x,
                   listdir(load_path)))
        init = files[0][:].find("discriminator") + 13
        epochs_ganTraining = files[0][init:-3]

        super().__init__(f"ganDiscriminator", f"{dataset_id}_{epochs_ganTraining}", epochs, batch_size)
        if 'version' in kwargs:
            ver=kwargs['version']
        if 'delete' in kwargs:
            supr= kwargs['delete']
        try:
            self.dataset = hd.load(dataset_id, version=ver, delete=supr)
        except:
            try:
                self.dataset=hd.load(dataset_id, version=ver)
            except:
                try:
                    self.dataset=hd.load(dataset_id, delete=supr)
                except:
                    self.dataset = hd.load(dataset_id)
        self.model_name = f"ganDiscriminator{epochs_ganTraining}"
        self.input_shape = self.dataset[0][0].shape
        self.dataset_id=dataset_id
        if (self.dataset_id == "indianA"):
            self.input_shape = (64, 64, self.input_shape[2])
        if (self.dataset_id == "indianB"):
            self.input_shape = (128, 128, self.input_shape[2])
        self.classes = self.dataset[1]['y'].max() + 1
        self.history = ""

    def get_loader(self)->Experiment:
        return ganDiscriminator()

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

        table = PrettyTable(["accuracy", "loss"])
        for i in range(len(loss_history)):
            table.add_row([acc_history[i], loss_history[i]])
        print(table)
        return acc_history[len(acc_history)-1] #return the last value

    def split(self, test_size):

        cant_examples = np.zeros(self.dataset[1]['y'].max() + 1)
        for i in self.dataset[1]['y']:
            cant_examples[i] += 1
        select = np.where(cant_examples >= (self.dataset[0].shape[0] / self.classes) * test_size)
        y_new = np.array((), dtype='uint8')
        pos = np.array((), dtype='uint8')
        for (k, cla) in enumerate(self.dataset[1]['y']):
            for j in select[0]:
                if (cla == j):
                    y_new = np.append(y_new, cla)
                    pos = np.append(pos, k)
        x_new = np.zeros((len(y_new), self.dataset[0].shape[1], self.dataset[0].shape[2], self.input_shape[2]), dtype='uint8')
        HEIGHT = 64
        WIDTH = 64

        if (self.dataset_id == "indianA"):
            X_new_resize = np.zeros((len(y_new), 64, 64, self.input_shape[2]))
        if (self.dataset_id == "indianB"):
            X_new_resize = np.zeros((len(y_new), 128, 128, 1))

        for (i, index) in enumerate(pos):
            x_new[i] = self.dataset[0][index]
            if (self.dataset_id == "indianA" or self.dataset_id == "indianB"):
                if (self.dataset_id == "indianA"):
                    image = transform.resize(x_new[i], (480, 640), preserve_range=True, mode="reflect",
                                             anti_aliasing=True)
                    image = Image.fromarray(image.astype(np.uint8), )

                    left = 20
                    top = 150.0
                    right = 550
                    bottom = 425.0
                    img = image.crop((left, top, right, bottom))
                    img2 = np.asarray(img)
                    X_new_resize[i] = transform.resize(img2, (HEIGHT, WIDTH), preserve_range=True, mode="reflect",
                                                       anti_aliasing=True)
                else:
                    X_new_resize[i] = transform.resize(x_new[i], (128, 128), preserve_range=True, mode="reflect",
                                                       anti_aliasing=True)
        if (self.dataset_id == "indianA" or self.dataset_id == "indianB"):
            X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X_new_resize, y_new,
                                                                                        test_size=test_size,
                                                                                        stratify=y_new)
        else:
            X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(x_new, y_new,
                                                                                        test_size=test_size,
                                                                                        stratify=y_new)
        if (X_train.shape[3]==1):
            X_train = np.repeat(X_train, 3, -1)
            X_test = np.repeat(X_test, 3, -1)

        X_train_preprocess = np.zeros((X_train.shape[0],X_train.shape[1],X_train.shape[2],X_train.shape[3]),
                                              dtype=np.float32)
        X_test_preprocess = np.zeros((X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3]),
                                              dtype=np.float32)
        for i, x in enumerate(X_train):
            x=tf.cast(x, tf.float32)
            x=keras.applications.mobilenet.preprocess_input(x)
            X_train_preprocess[i] = x
        for j, xt in enumerate(X_test):
            xt = tf.cast(xt, tf.float32)
            xt = keras.applications.mobilenet.preprocess_input(xt)
            X_test_preprocess[j] = xt

        return X_train_preprocess, X_test_preprocess, Y_train, Y_test

    def build_model(self):
        img = keras.layers.Input(shape=(self.input_shape[0],self.input_shape[1],3))
        load_path = os.path.join(gan_folder,self.dataset_id)
        files = list(
            filter(lambda x: ".h5" in x,
                   listdir(load_path)))
        prev_model = load_model(os.path.join(load_path, files[0]))
        prev_model.load_weights(os.path.join(load_path, files[1]))

        top_model = keras.models.Sequential()
        top_model.add(Dense(self.classes, activation='softmax'))

        model = Model(inputs=img, outputs=top_model(prev_model(img)))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

        return model


    def graphics(self, model, X_test, y_true, show_graphic, show_matrix):
        path = self.get_path()
        graphic_acc_file=os.path.join(path,"figure_acc.png")
        graphic_loss_file = os.path.join(path, "figure_loss.png")
        self.plot_training_curves(self.history,graphic_acc_file, graphic_loss_file,show_graphic)

        if(self.classes>60):
            print("It takes some minutes, because of the amount of classes")
            print("Maybe you should make zoom in the save file")

        graphic_matrix = os.path.join(path, "matrix_confusion.png")
        y_pred = model.predict(X_test)
        self.plot_confusion_matrix(y_true, np.argmax(y_pred, axis = 1),graphic_matrix,show_matrix)
import numpy as np
import keras
from keras.models import Model
import sklearn
from skimage import transform
import tensorflow as tf
from keras.models import load_model
import handshape_datasets as hd
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

        super().__init__("ganDiscriminator", dataset_id, epochs, batch_size)
        self.tl = True
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
        self.model_name = "ganDiscriminator"
        self.input_shape = self.dataset[0][0].shape

        HEIGHT=128
        WIDTH=128

        if (self.input_shape[0]+self.input_shape[1]>600):
            self.input_shape=(HEIGHT,WIDTH,self.input_shape[2])
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
        for (i, index) in enumerate(pos):
            x_new[i] = self.dataset[0][index]
        X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(x_new, y_new,
                                                                                    test_size=test_size,
                                                                                    stratify=y_new)

        if(X_train.shape[1]+X_train.shape[2] > 600):

            HEIGHT=128
            WIDTH=128
            X_train_resize = np.zeros((X_train.shape[0],HEIGHT,WIDTH,1))
            X_test_resize = np.zeros((X_test.shape[0], HEIGHT, WIDTH, 1))
            for i,x in enumerate(X_train):
                X_train_resize[i]=transform.resize(
                    x, (HEIGHT, WIDTH), preserve_range=True, mode="reflect", anti_aliasing=True)
            for i,x_t in enumerate(X_test):
                X_test_resize[i]=transform.resize(
                    x_t, (HEIGHT, WIDTH), preserve_range=True, mode="reflect", anti_aliasing=True)

            if (X_train_resize.shape[3] == 1):
                X_train_resize = np.repeat(X_train_resize, 3, -1)
                X_test_resize = np.repeat(X_test_resize, 3, -1)

            X_train_resize_preprocess = np.zeros((X_train_resize.shape[0], X_train_resize.shape[1], X_train_resize.shape[2], X_train_resize.shape[3]),
                                          dtype=np.float32)
            X_test_resize_preprocess = np.zeros((X_test_resize.shape[0], X_test_resize.shape[1], X_test_resize.shape[2], X_test_resize.shape[3]),
                                         dtype=np.float32)
            for i, x in enumerate(X_train_resize):
                x = tf.cast(x, tf.float32)
                x = keras.applications.mobilenet.preprocess_input(x)
                X_train_resize_preprocess[i] = x
            for j, xt in enumerate(X_test_resize):
                xt = tf.cast(xt, tf.float32)
                xt = keras.applications.mobilenet.preprocess_input(xt)
                X_test_resize_preprocess[j] = xt
            return X_train_resize_preprocess, X_test_resize_preprocess, Y_train, Y_test
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

        newshape = X_train.shape[1] * X_train.shape[2] * X_train.shape[3]
        X_train_preprocess = np.reshape(X_train_preprocess, (X_train.shape[0], newshape))
        X_test_preprocess = np.reshape(X_test_preprocess, (X_test.shape[0], newshape))

        return X_train_preprocess, X_test_preprocess, Y_train, Y_test

    def build_model(self):

        #model = load_model(os.path.join(gan_folder,"GANdiscriminator.h5"))

        model = keras.models.Sequential(name="discriminator")
        model.add(keras.layers.Dense(512, input_dim=np.prod(self.input_shape)))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.Dense(512))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.Dropout(0.4))
        model.add(keras.layers.Dense(512))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.Dropout(0.4))
        model.add(keras.layers.Dense(self.classes, activation='sigmoid'))

        #model.add(keras.layers.Dense(self.classes, activation='softmax'))

        """
        top_model = keras.models.Sequential()
        top_model.add(ZeroPadding2D((1, 1), input_shape=self.input_shape))
        top_model.add(Conv2D(64, (3, 3), activation='relu'))
        top_model.add(ZeroPadding2D((1, 1)))
        top_model.add(Conv2D(64, (3, 3), activation='relu'))
        top_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        top_model.add(ZeroPadding2D((1, 1)))
        top_model.add(Conv2D(128, (3, 3), activation='relu'))
        top_model.add(ZeroPadding2D((1, 1)))
        top_model.add(Conv2D(128, (3, 3), activation='relu'))
        top_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        top_model.add(ZeroPadding2D((1, 1)))
        top_model.add(Conv2D(256, (3, 3), activation='relu'))
        top_model.add(ZeroPadding2D((1, 1)))
        top_model.add(Conv2D(256, (3, 3), activation='relu'))
        top_model.add(ZeroPadding2D((1, 1)))
        top_model.add(Conv2D(256, (3, 3), activation='relu'))
        top_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        top_model.add(ZeroPadding2D((1, 1)))
        top_model.add(Conv2D(512, (3, 3), activation='relu'))
        top_model.add(ZeroPadding2D((1, 1)))
        top_model.add(Conv2D(512, (3, 3), activation='relu'))
        top_model.add(ZeroPadding2D((1, 1)))
        top_model.add(Conv2D(512, (3, 3), activation='relu'))
        top_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        top_model.add(ZeroPadding2D((1, 1)))
        top_model.add(Conv2D(512, (3, 3), activation='relu'))
        top_model.add(ZeroPadding2D((1, 1)))
        top_model.add(Conv2D(512, (3, 3), activation='relu'))
        top_model.add(ZeroPadding2D((1, 1)))
        top_model.add(Conv2D(512, (3, 3), activation='relu'))
        top_model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        top_model.add(Flatten())

        top_model.add(Dense(4096, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(self.classes))
        top_model.add(Activation('softmax'))
        """
        #model.load_weights(os.path.join(gan_folder, "GANdiscriminator_weights.h5"))
        #model.add(keras.layers.Reshape(self.input_shape))


        #model.summary()


        #model.summary()

        model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
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
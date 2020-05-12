import numpy as np
import keras
from keras.models import Model
import sklearn
import handshape_datasets as hd
import os
from experiment import Experiment
from prettytable import PrettyTable
from sklearn import model_selection

class MobileNet(Experiment):

    def __init__(self, epochs, batch_size, dataset_id, **kwargs):
        super().__init__("MobileNet", dataset_id, epochs, batch_size)
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
        x_new = np.zeros((len(y_new), self.input_shape[0], self.input_shape[1], self.input_shape[2]), dtype='uint8')
        for (i, index) in enumerate(pos):
            x_new[i] = self.dataset[0][index]
        X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(x_new, y_new,
                                                                                    test_size=test_size,
                                                                                    stratify=y_new)
        if (X_train.shape[3]==1):
            X_train = np.repeat(X_train, 3, -1)
            X_test = np.repeat(X_test, 3, -1)
        return X_train, X_test, Y_train, Y_test

    def build_model(self):
        base_model = keras.applications.mobilenet.MobileNet(input_shape=(self.input_shape[0],self.input_shape[1],3), weights='imagenet',
                                                            include_top=False)
        output = keras.layers.GlobalAveragePooling2D()(base_model.output)
        output = keras.layers.Dense(32, activation='relu')(output)
        # Nueva capa de salida
        output = keras.layers.Dense(self.classes, activation='softmax')(output)  # cambiar cantidad de clases
        model = Model(inputs=base_model.input, outputs=output)
        # Entrenar con nuevos datos
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


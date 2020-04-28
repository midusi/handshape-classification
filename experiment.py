from abc import abstractmethod, ABC
from pathlib import Path
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import logging


default_folder = Path.home() / 'handshape_classification' / 'Results'

class Experiment(ABC):

    def __init__(self, model, dataset, epochs, batch_size):
        self.id=f"{dataset}_{model}_batch{batch_size}_epochs{epochs}"
        self.epochs=epochs
        self.batch_size=batch_size
        self.model=model
        self.dataset=dataset
        self.path = os.path.join(default_folder, self.id)
        if not os.path.exists(self.path):
            logging.info("Create folder")
            os.makedirs(self.path)

    @property
    @abstractmethod
    def get_result(self):
        pass

    @abstractmethod
    def get_params(self):
        pass

    def get_path(self):
        return self.path


    def get_id(self):
        return self.id

    def plot_training_curves(self, history, graphic_acc, graphic_loss, acc=True):
        # summarize history for accuracy
        if (acc):
            plt.figure()
            plt.grid()
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.savefig(graphic_acc, dpi=300)
            plt.show()

        # summarize history for loss
        plt.figure()
        plt.grid()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(graphic_loss, dpi=300)
        plt.show()
        return True

    # Crea y Grafica una matriz de confusión
    # PARAM:
    #       real_target = vector con valores esperados
    #       pred_target = vector con valores calculados por un modelo
    #       classes = lista de strings con los nombres de las clases.
    def plot_confusion_matrix(self, real_target, pred_target, classes=[], normalize=False, title='Confusion matrix',
                              cmap=plt.cm.Blues):


        if (len(classes) == 0):
            classes = [str(i) for i in range(int(max(real_target) + 1))]  # nombres de clases consecutivos
        cm = confusion_matrix(real_target, pred_target)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        #    plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    @abstractmethod
    def load(self):

        pass

    @abstractmethod
    def split(self):

        pass

    @abstractmethod
    def get_loader(self):

        pass

    @abstractmethod
    def build_model(self):

        pass

    @abstractmethod
    def graphics(self):

        pass

    @abstractmethod
    def get_history(self):

        pass
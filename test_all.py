
from Experiments import mobile_net as mn
from Experiments import dense_net as dn
from Experiments import efficient_net as en
import handshape_datasets as hd
import numpy as np
from pathlib import Path
import os

from prettytable import PrettyTable
import parameters

epochs=15
iteracion=1

default_folder = Path.home() / 'handshape-classification' / 'Results'

for dataset_id in hd.ids():
    acc_avg_mo=np.zeros(iteracion)
    acc_avg_de=np.zeros(iteracion)
    acc_avg_eff=np.zeros(iteracion)

    for i in range(iteracion):

        # MobileNet
        mobile = mn.MobileNet(epochs, parameters.get_batch_mobile(dataset_id), dataset_id)
        model_mo = mobile.build_model()
        X_train, X_test, Y_train, Y_test = mobile.split(parameters.get_split_value(dataset_id))
        history = mobile.load(model_mo, X_train, Y_train, X_test, Y_test)
        mobile.graphics(model_mo, X_test, Y_test,show_graphic=True, show_matrix=True)
        acc_last_mobile=mobile.get_result()
        acc_avg_mo[i]=acc_last_mobile

        #DenseNet
        denseNet = dn.DenseNet(epochs, parameters.get_batch_dense(dataset_id), dataset_id)
        model_de = denseNet.build_model()
        X_train, X_test, Y_train, Y_test = denseNet.split(parameters.get_split_value(dataset_id))
        history = denseNet.load(model_de, X_train, Y_train, X_test, Y_test)
        denseNet.graphics(model_de, X_test, Y_test,show_graphic=True, show_matrix=True)
        acc_last_dense=denseNet.get_result()
        acc_avg_de[i] = acc_last_dense

        # EfficientNet
        efficientNet = en.EfficientNet(epochs, parameters.get_batch_eff(dataset_id), dataset_id)
        model_eff = efficientNet.build_model()
        X_train, X_test, Y_train, Y_test = efficientNet.split(parameters.get_split_value(dataset_id))
        history = efficientNet.load(model_eff, X_train, Y_train, X_test, Y_test)
        efficientNet.graphics(model_eff, X_test, Y_test, show_graphic=True, show_matrix=True)
        acc_last_eff = efficientNet.get_result()
        acc_avg_eff[i] = acc_last_eff

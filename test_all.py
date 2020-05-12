
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

#table=PrettyTable(["Dataset", "MobileNet", "DenseNet","EfficientNet"])
for dataset_id in hd.ids():
    acc_avg_mo=np.zeros(iteracion)
    acc_avg_de=np.zeros(iteracion)
    acc_avg_eff=np.zeros(iteracion)

    for i in range(iteracion):
        """
        # MobileNet
        mobile = mn.MobileNet(epochs, parameters.get_batch_mobile(dataset_id), dataset_id)
        model = mobile.build_model()
        X_train, X_test, Y_train, Y_test = mobile.split(parameters.get_split_value(dataset_id))
        history = mobile.load(model, X_train, Y_train, X_test, Y_test,show_graphic=True, show_matrix=True)
        mobile.graphics(model, X_test, Y_test)
        acc_last_mobile=mobile.get_result()
        acc_avg_mo[i]=acc_last_mobile

        #DenseNet
        denseNet = dn.DenseNet(epochs, parameters.get_batch_dense(dataset_id), dataset_id)
        model = denseNet.build_model()
        X_train, X_test, Y_train, Y_test = denseNet.split(parameters.get_split_value(dataset_id))
        history = denseNet.load(model, X_train, Y_train, X_test, Y_test,show_graphic=True, show_matrix=True)
        denseNet.graphics(model, X_test, Y_test)
        acc_last_dense=denseNet.get_result()
        acc_avg_de[i] = acc_last_dense
"""
        # EfficientNet
        efficientNet = en.EfficientNet(epochs, parameters.get_batch_eff(dataset_id), dataset_id)
        model = efficientNet.build_model()
        X_train, X_test, Y_train, Y_test = efficientNet.split(parameters.get_split_value(dataset_id))
        history = efficientNet.load(model, X_train, Y_train, X_test, Y_test)
        efficientNet.graphics(model, X_test, Y_test, show_graphic=False, show_matrix=False)
        acc_last_eff = efficientNet.get_result()
        acc_avg_eff[i] = acc_last_eff

    #table.add_row([dataset_id,acc_avg_mo.mean(), acc_avg_de.mean(),acc_avg_eff.mean()])
    #print("Accuracy values:")
    #print(table)
#data = table.get_string()
#print(data)
#file = os.path.join(default_folder, 'Accuracy_table.txt')
#with open(file, 'w') as f:
#    f.write(data)
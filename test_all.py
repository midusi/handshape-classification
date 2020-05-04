import experiment
from Experiments import mobile_net as mn
from Experiments import dense_net as dn
import handshape_datasets as hd
import numpy as np

from prettytable import PrettyTable
import parameters

epochs=15
#batch_size=32


table=PrettyTable(["Dataset", "MobileNet", "DenseNet"])
for dataset_id in hd.ids():
    acc_avg_mo=np.zeros(10)
    acc_avg_de=np.zeros(10)

    for i in range(10):

        # MobileNet
        mobile = mn.MobileNet(epochs, parameters.get_batch_mobile(dataset_id), dataset_id)
        model = mobile.build_model()
        X_train, X_test, Y_train, Y_test = mobile.split(parameters.get_split_value(dataset_id))
        history = mobile.load(model, X_train, Y_train, X_test, Y_test)
        mobile.graphics(model, X_test, Y_test)
        acc_last_mobile, acc_avg_mobile=mobile.get_result()

        acc_avg_mo[i]=acc_avg_mobile

        #DenseNet
        denseNet = dn.DenseNet(epochs, parameters.get_batch_dense(dataset_id), dataset_id)
        model = denseNet.build_model()
        X_train, X_test, Y_train, Y_test = denseNet.split(parameters.get_split_value(dataset_id))
        history = denseNet.load(model, X_train, Y_train, X_test, Y_test)
        denseNet.graphics(model, X_test, Y_test)
        acc_last_dense, acc_avg_dense=denseNet.get_result()

        acc_avg_de[i] = acc_avg_dense

    table.add_row(dataset_id,acc_avg_mo.mean(), acc_avg_de.mean())
    print("Accuracy values:")
    print(table)
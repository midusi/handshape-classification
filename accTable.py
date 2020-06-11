from os import listdir

from pathlib import Path
from prettytable import PrettyTable
import os
import parameters
import handshape_datasets as hd
import numpy as np


default_folder = Path.home() / 'handshape-classification' / 'Results'
folders = ["MobileNet", "DenseNet", "EfficientNet"]
epochs = 15
table=PrettyTable(["Dataset", "MobileNet", "DenseNet", "EfficientNet"])
table_test=PrettyTable(["Dataset", "MobileNet", "DenseNet", "EfficientNet"])

table_noTL=PrettyTable(["Dataset", "MobileNet", "DenseNet", "EfficientNet"])
table_test_noTL=PrettyTable(["Dataset", "MobileNet", "DenseNet", "EfficientNet"])

for dataset_id in hd.ids():
    acc_value=np.zeros((len(folders)))
    val_acc_value = np.zeros((len(folders)))

    acc_value_noTL = np.zeros((len(folders)))
    val_acc_value_noTL = np.zeros((len(folders)))
    for i,model in enumerate(folders):
        model_path = os.path.join(default_folder, model)
        if (model=="MobileNet"):
            batch_size=parameters.get_batch_mobile(dataset_id)
        else:
            if(model=="DenseNet"):
                batch_size = parameters.get_batch_dense(dataset_id)
            else:
                batch_size = parameters.get_batch_eff(dataset_id)
        subsets_folders = list(
            filter(lambda x: f"{dataset_id}_{model}_batch{batch_size}_epochs{epochs}" in x,
                listdir(model_path)))
        subsets_folders_TL= list(
            filter(lambda x: f"_noTL" not in x,
                subsets_folders))

        subsets_folders_noTL = list(
            filter(lambda x: f"{dataset_id}_{model}_batch{batch_size}_epochs{epochs}_noTL" in x,
                   listdir(model_path)))

        folder_act_noTL = os.path.join(model_path, subsets_folders_noTL[0])
        acc_history_path_noTL = os.path.join(folder_act_noTL, "acc_history.txt")
        val_acc_history_path_noTL = os.path.join(folder_act_noTL, "val_acc_history.txt")
        with open(acc_history_path_noTL) as f:
            lines = f.readlines()
            acc_value_noTL[i] = lines[len(lines) - 1]
        with open(val_acc_history_path_noTL) as f:
            lines = f.readlines()
            val_acc_value_noTL[i] = lines[len(lines) - 1]

        folder_act = os.path.join(model_path, subsets_folders_TL[0])
        acc_history_path = os.path.join(folder_act, "acc_history.txt")
        val_acc_history_path = os.path.join(folder_act, "val_acc_history.txt")
        with open(acc_history_path) as f:
            lines = f.readlines()
            acc_value[i]=lines[len(lines)-1]
        with open(val_acc_history_path) as f:
            lines = f.readlines()
            val_acc_value[i]=lines[len(lines)-1]

    table_test_noTL.add_row([dataset_id, val_acc_value_noTL[0], val_acc_value_noTL[1], val_acc_value_noTL[2]])
    table_noTL.add_row([dataset_id, acc_value_noTL[0], acc_value_noTL[1], acc_value_noTL[2]])

    table_test.add_row([dataset_id, val_acc_value[0], val_acc_value[1], val_acc_value[2]])
    table.add_row([dataset_id, acc_value[0], acc_value[1], acc_value[2]])
print (table)
print (table_test)

print (table_noTL)
print (table_test_noTL)

data = table.get_string()
data_test= table_test.get_string()

data_noTL = table_noTL.get_string()
data_test_noTL= table_test_noTL.get_string()

file_noTL = os.path.join(default_folder, 'Accuracy_table_noTL.txt')
with open(file_noTL, 'w') as f:
   f.write(data_noTL)

file_test_noTL = os.path.join(default_folder, 'val_test_Accuracy_table_noTL.txt')
with open(file_test_noTL, 'w') as f:
   f.write(data_test_noTL)

file = os.path.join(default_folder, 'Accuracy_table.txt')
with open(file, 'w') as f:
   f.write(data)

file_test = os.path.join(default_folder, 'val_test_Accuracy_table.txt')
with open(file_test, 'w') as f:
   f.write(data_test)





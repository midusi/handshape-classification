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

for dataset_id in hd.ids():
    acc_value=np.zeros((len(folders)))
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
        subsets_folders_noTL= list(
            filter(lambda x: f"_noTL" not in x,
                subsets_folders))
        folder_act = os.path.join(model_path, subsets_folders_noTL[0])
        acc_history_path = os.path.join(folder_act, "acc_history.txt")

        with open(acc_history_path) as f:
            lines = f.readlines()
            acc_value[i]=lines[len(lines)-1]

    table.add_row([dataset_id, acc_value[0], acc_value[1], acc_value[2]])
print (table)
data = table.get_string()
file = os.path.join(default_folder, 'Accuracy_table.txt')
with open(file, 'w') as f:
   f.write(data)





import experiment
from Experiments import mobile_net as mn
from Experiments import dense_net as dn
from Experiments import efficient_net as en
from Experiments import ganDiscriminator as gd
import handshape_datasets as hd
import parameters
import numpy as np
from prettytable import PrettyTable
from pathlib import Path
import math

epochs=15
dataset_id="lsa16"
iteracion=1
showgraphics=True
transferl=True

default_folder = Path.home() / 'handshape-classification' / 'Results'
acc_avg_eff=np.zeros(iteracion)
acc_avg_mo=np.zeros(iteracion)
acc_avg_de=np.zeros(iteracion)
acc_avg_gd=np.zeros(iteracion)

for i in range(iteracion):
  """
  #MobileNet
  mobile = mn.MobileNet(epochs, parameters.get_batch_mobile(dataset_id), dataset_id,tl=transferl)
  model = mobile.build_model()
  X_train, X_test, Y_train, Y_test = mobile.split(parameters.get_split_value(dataset_id))
  history = mobile.load(model, X_train, Y_train, X_test, Y_test)
  mobile.graphics(model, X_test, Y_test, show_graphic=showgraphics, show_matrix=showgraphics)
  acc_last_mobile=mobile.get_result()
  acc_avg_mo[i]=acc_last_mobile

  #DenseNet
  denseNet = dn.DenseNet(epochs, parameters.get_batch_dense(dataset_id), dataset_id,tl=transferl)
  model = denseNet.build_model()
  X_train, X_test, Y_train, Y_test = denseNet.split(parameters.get_split_value(dataset_id))
  history = denseNet.load(model, X_train, Y_train, X_test, Y_test)
  denseNet.graphics(model, X_test, Y_test, show_graphic=showgraphics, show_matrix=showgraphics)
  acc_last_dense=denseNet.get_result()
  acc_avg_de[i] = acc_last_dense

  #EfficientNet
  efficientNet = en.EfficientNet(epochs, parameters.get_batch_eff(dataset_id), dataset_id,tl=transferl)
  model = efficientNet.build_model()
  X_train, X_test, Y_train, Y_test = efficientNet.split(parameters.get_split_value(dataset_id))
  history = efficientNet.load(model, X_train, Y_train, X_test, Y_test)
  efficientNet.graphics(model, X_test, Y_test, show_graphic=showgraphics, show_matrix=showgraphics)
  acc_last_eff = efficientNet.get_result()
  acc_avg_eff[i] = acc_last_eff
  """
  # GanDiscriminator
  gdm = gd.ganDiscriminator(epochs, parameters.get_batch_mobile(dataset_id), dataset_id, tl=transferl)
  gdmodel = gdm.build_model()
  X_train, X_test, Y_train, Y_test = gdm.split(parameters.get_split_value(dataset_id))
  history = gdm.load(gdmodel, X_train, Y_train, X_test, Y_test)
  gdm.graphics(gdmodel, X_test, Y_test, show_graphic=showgraphics, show_matrix=showgraphics)
  acc_last_gd = gdm.get_result()
  acc_avg_gd[i] = acc_last_gd





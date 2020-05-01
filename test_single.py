import experiment
from Experiments import mobile_net as mn
from Experiments import dense_net as dn
import handshape_datasets as hd

epochs=15
batch_size=64
dataset_id="PugeaultASL_A"

#
# MobileNet
mobile = mn.MobileNet(epochs, batch_size, dataset_id)
model = mobile.build_model()
X_train, X_test, Y_train, Y_test = mobile.split(0.3)
history = mobile.load(model, X_train, Y_train, X_test, Y_test)
mobile.graphics(model, X_test, Y_test)
mobile.get_result()
"""

#DenseNet
denseNet = dn.DenseNet(epochs, batch_size, dataset_id)
model = denseNet.build_model()
X_train, X_test, Y_train, Y_test = denseNet.split(0.25)
history = denseNet.load(model, X_train, Y_train, X_test, Y_test)
denseNet.graphics()
denseNet.get_result()
"""
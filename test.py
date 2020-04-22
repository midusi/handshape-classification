import experiment
from Experiments import mobile_net as mn

epochs=20
batch_size=32
dataset_id="lsa16"
mobile=mn.MobileNet(epochs, batch_size, dataset_id)
model=mobile.build_model()
X_train, X_test, Y_train, Y_test=mobile.split(0.1)
history=mobile.load(model, X_train, Y_train, X_test, Y_test)
mobile.graphics(history)
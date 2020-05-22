from keras.datasets import mnist
from pathlib import Path
from tensorflow.keras.optimizers import Adam
import numpy as np
import keras.layers
from keras.models import Model
import matplotlib.pyplot as plt
import sklearn
import handshape_datasets as hd
import parameters
import os
from sklearn import model_selection
import handshape_datasets

default_folder = Path.home() / 'handshape-classification' / 'GANResults'

class GAN():

    def __init__(self,dataset_id,**kwargs):
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

        self.input_shape = self.dataset[0][0].shape
        self.img_rows = self.input_shape[0]
        self.img_cols = self.input_shape[1]
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.name=dataset_id

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generated imgs
        z = keras.layers.Input(shape=(100,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        self.path = default_folder
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def split(self, test_size,x,y):

        cant_examples = np.zeros(y.max() + 1)
        classes = y.max() + 1
        input_shape = x[0].shape

        for i in y:
            cant_examples[i] += 1
        select = np.where(cant_examples >= (x.shape[0] / classes) * test_size)
        y_new = np.array((), dtype='uint8')
        pos = np.array((), dtype='uint8')
        for (k, cla) in enumerate(y):
            for j in select[0]:
                if (cla == j):
                    y_new = np.append(y_new, cla)
                    pos = np.append(pos, k)
        x_new = np.zeros((len(y_new), input_shape[0], input_shape[1], input_shape[2]), dtype='uint8')
        for (i, index) in enumerate(pos):
            x_new[i] = x[index]
        X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(x_new, y_new,
                                                                                    test_size=test_size,
                                                                                    stratify=y_new)
        if (X_train.shape[3] == 1):
            X_train = np.repeat(X_train, 3, -1)
            X_test = np.repeat(X_test, 3, -1)

        return X_train, X_test, Y_train, Y_test

    def build_generator(self):

        noise_shape = (100,)

        model = keras.models.Sequential()

        model.add(keras.layers.Dense(256, input_shape=noise_shape))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.BatchNormalization(momentum=0.8))
        model.add(keras.layers.Dense(512))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.BatchNormalization(momentum=0.8))
        model.add(keras.layers.Dense(1024))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.BatchNormalization(momentum=0.8))
        model.add(keras.layers.Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(keras.layers.Reshape(self.img_shape))

        model.summary()

        noise = keras.layers.Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        img_shape = (self.img_rows, self.img_cols, self.channels)

        model = keras.models.Sequential()

        model.add(keras.layers.Flatten(input_shape=img_shape))
        model.add(keras.layers.Dense(512))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.Dense(256))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.Dense(1, activation='sigmoid'))
        model.summary()

        img = keras.layers.Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, dataset_id,epochs, batch_size=128, save_interval=50):

        # Load the dataset
        #(X_train, _), (_, _) = mnist.load_data()
        x,metadata= hd.load(dataset_id)
        X_train, X_test, Y_train, Y_test=self.split(parameters.get_split_value(dataset_id),x,metadata['y'])
        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        #X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (half_batch, 100))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 100))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        save_path=os.path.join(self.path,self.name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fig.savefig(os.path.join(save_path, f"GANimage_{epoch}.png"))
        plt.close()

"""
if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=30000, batch_size=32, save_interval=200)
"""
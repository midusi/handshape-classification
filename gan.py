import math
from PIL import Image

from keras.datasets import mnist
from pathlib import Path
from skimage import transform
from tensorflow.keras.optimizers import Adam
import numpy as np
import keras.layers
from keras.models import Model
import matplotlib.pyplot as plt
import sklearn
import handshape_datasets as hd
import parameters
from keras.models import load_model
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

        self.img_rows = (self.input_shape[0] // 4) * 4
        self.img_cols = (self.input_shape[1] // 4) * 4

        self.channels = 3
        self.name = dataset_id
        if(self.name=="psl" or self.name=="indianB"):
            self.img_shape=(128,128,self.channels)
            self.img_rows = 128
            self.img_cols = 128
        else:
            if(self.name=="indianA"):
                self.img_shape=(64,64,self.channels)
                self.img_rows = 64
                self.img_cols = 64
            else:
                self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.classes = self.dataset[1]['y'].max() + 1
        self.noise_value=100


        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.base_model, self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        noise = keras.layers.Input(shape=(self.noise_value,))
        label = keras.layers.Input(shape=(1,))
        img = self.generator([noise, label])

        #self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generated imgs
        #z = keras.layers.Input(shape=(100,))
        #img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator([img, label])

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model([noise, label], valid)
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
        x_new = np.zeros((len(y_new), input_shape[0], input_shape[1], self.channels), dtype='uint8')

        if (self.name == "indianA"):
            X_new_resize = np.zeros((len(y_new), 64, 64, self.input_shape[2]))
        if (self.name == "indianB"):
            X_new_resize = np.zeros((len(y_new), 128, 128, self.channels))
        if (self.name == "psl"):
            X_new_resize = np.zeros((len(y_new), 128, 128, self.channels))


        for (i, index) in enumerate(pos):
            x_new[i] = self.dataset[0][index]
            if (self.name == "indianA" or self.name == "indianB" or self.name=="psl"):
                if (self.name == "indianA"):
                    image = transform.resize(x_new[i], (480, 640), preserve_range=True, mode="reflect",
                                             anti_aliasing=True)
                    image = Image.fromarray(image.astype(np.uint8), )

                    left = 20
                    top = 150.0
                    right = 550
                    bottom = 425.0
                    img = image.crop((left, top, right, bottom))
                    img2 = np.asarray(img)
                    X_new_resize[i] = transform.resize(img2, (64, 64), preserve_range=True, mode="reflect",
                                                       anti_aliasing=True)
                else:
                    X_new_resize[i] = transform.resize(x_new[i], (128, 128), preserve_range=True, mode="reflect",
                                                       anti_aliasing=True)
            else:
                if ((x[index].shape[0] % 4 != 0) or (x[index].shape[1] % 4 != 0)):
                    x_new[i] = transform.resize(x[index], (self.img_rows, self.img_cols), preserve_range=True,
                                                mode="reflect",
                                                anti_aliasing=True)
                else:
                    x_new[i] = x[index]
        if (self.name == "indianA" or self.name == "indianB" or self.name=="psl"):
            X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X_new_resize, y_new,
                                                                                            test_size=test_size,
                                                                                            stratify=y_new)
        else:
            X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(x_new, y_new,
                                                                                            test_size=test_size,
                                                                                            stratify=y_new)


        if (X_train.shape[3] == 1):
            X_train = np.repeat(X_train, 3, -1)
            X_test = np.repeat(X_test, 3, -1)

        return X_train, X_test, Y_train, Y_test

    def build_generator(self):

        noise_shape=(self.noise_value,)


        model = keras.models.Sequential(name='generator')
        h, w, c = self.img_rows, self.img_cols, self.channels
        filters = 256 #256
        # Imagen inicial de 7x7 (asumo que genero algo de 28x28)
        image_dim = filters * (h // 4) * (w // 4)
        model.add(keras.layers.Dense(image_dim, input_shape=noise_shape))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.Reshape((h // 4, w // 4, filters)))
        model.add(keras.layers.Dense(256))
        # Convertir a 14x14
        model.add(keras.layers.Conv2DTranspose(filters, (4, 4), strides=(2, 2), padding='same'))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.Dense(512))
        # Convertir a 28x28
        model.add(keras.layers.Conv2DTranspose(filters, (4, 4), strides=(2, 2), padding='same'))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        #model.add(keras.layers.Dense(np.prod(self.img_shape), activation='tanh'))
        # Imagen final de 28x28x1
        model.add(keras.layers.Conv2D(c, (7, 7), activation='tanh', padding='same'))
        """
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
        """

        model.summary()
        noise = keras.layers.Input(shape=(self.noise_value,))
        label = keras.layers.Input(shape=(1,), dtype='int32')
        label_embedding = keras.layers.Flatten()(keras.layers.Embedding(self.classes, self.noise_value)(label))

        model_input = keras.layers.multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self):

        model = keras.models.Sequential(name="discriminator")

        model.add(keras.layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same'))
        model.add(keras.layers.LeakyReLU())
        model.add(keras.layers.Dense(512))
        model.add(keras.layers.Dropout(0.3))
    
        model.add(keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
        model.add(keras.layers.LeakyReLU())
        model.add(keras.layers.Dense(256))
        model.add(keras.layers.Dropout(0.3))

        model.add(keras.layers.GlobalAveragePooling2D())

        """

        model = keras.models.Sequential(name="discriminator")

        model.add(keras.layers.Dense(512, input_dim=np.prod(self.img_shape)))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.Dense(512))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.Dropout(0.4))
        model.add(keras.layers.Dense(512))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Dense(1, activation='sigmoid'))
        """

        img = keras.layers.Input(shape=self.img_shape)
        label = keras.layers.Input(shape=(1,), dtype='int32')

        label_embedding = keras.layers.Flatten()(keras.layers.Embedding(self.classes, np.prod(self.img_shape))(label))
        flat_img = keras.layers.Flatten()(img)

        model_input = keras.layers.multiply([flat_img, label_embedding])
        model_input = keras.layers.Reshape(self.img_shape)(model_input)

        validity = model(model_input)
        discriminator=keras.layers.Dense(1, activation='sigmoid')(validity)
        discriminator=Model(inputs=[img,label],outputs=discriminator)

        return model, discriminator

    def train(self, dataset_id,epochs, batch_size=128, save_interval=50):

        # Load the dataset
        #(X_train, _), (_, _) = mnist.load_data()
        x,metadata= hd.load(dataset_id)
        X_train, X_test, Y_train, Y_test=self.split(parameters.get_split_value(dataset_id),x,metadata['y'])
        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        #X_train = np.expand_dims(X_train, axis=3)

        #half_batch = int(batch_size / 2)

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs, labels = X_train[idx], Y_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.noise_value))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, labels])

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            sampled_labels = np.random.randint(0, self.classes, batch_size).reshape(-1, 1)

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            #valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)
        save_path = os.path.join(default_folder, self.name)
        self.base_model.save(os.path.join(save_path,f"{self.name}_GANdiscriminator{epochs}.h5"))
        self.base_model.save_weights(os.path.join(save_path,f"{self.name}_GANdiscriminator{epochs}_weights.h5"))
        print("Saved model to disk")

    def save_imgs(self, epoch):
        sqr_classes = math.sqrt(self.classes)
        if (sqr_classes % 1 > 0):
            sqr_classes = int(sqr_classes) + 1
        else:
            sqr_classes = int(sqr_classes)

        r, c = int(sqr_classes), int(sqr_classes)
        noise = np.random.normal(0, 1, (r*c, self.noise_value))
        sampled_labels = np.arange(0, r*c).reshape(-1, 1)
        gen_imgs = self.generator.predict([noise, sampled_labels])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:])
                axs[i,j].axis('off')
                cnt += 1
                if(cnt>self.classes):
                    axs[i, j].set_visible(False)
            if (cnt > self.classes):
                axs[i, j].set_visible(False)
        save_path=os.path.join(self.path,self.name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fig.savefig(os.path.join(save_path, f"GANimage_{epoch}.png"))
        plt.close()

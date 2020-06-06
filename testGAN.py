from gan import GAN
import handshape_datasets as hd
from fastGan import FastSRGAN
import tensorflow as tf

dataset_id="Nus1"

gan = GAN(dataset_id)
gan.train(dataset_id,epochs=30000, batch_size=32, save_interval=200)


# Parse the CLI arguments.
"""
def train_step(model, x, y):
    Single train step function for the SRGAN.
    Args:
        model: An object that contains a tf keras compiled discriminator model.
        x: The low resolution input image.
        y: The desired high resolution output image.
    Returns:
        d_loss: The mean loss of the discriminator.
    
    # Label smoothing for better gradient flow
    valid = tf.ones((x.shape[0],) + model.disc_patch)
    fake = tf.zeros((x.shape[0],) + model.disc_patch)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # From low res. image generate high res. version
        fake_hr = model.generator(x)

        # Train the discriminators (original images = real / generated = Fake)
        valid_prediction = model.discriminator(y)
        fake_prediction = model.discriminator(fake_hr)

        # Generator loss
        content_loss = model.content_loss(y, fake_hr)
        adv_loss = 1e-3 * tf.keras.losses.BinaryCrossentropy()(valid, fake_prediction)
        mse_loss = tf.keras.losses.MeanSquaredError()(y, fake_hr)
        perceptual_loss = content_loss + adv_loss + mse_loss

        # Discriminator loss
        valid_loss = tf.keras.losses.BinaryCrossentropy()(valid, valid_prediction)
        fake_loss = tf.keras.losses.BinaryCrossentropy()(fake, fake_prediction)
        d_loss = tf.add(valid_loss, fake_loss)

    # Backprop on Generator
    gen_grads = gen_tape.gradient(perceptual_loss, model.generator.trainable_variables)
    model.gen_optimizer.apply_gradients(zip(gen_grads, model.generator.trainable_variables))

    # Backprop on Discriminator
    disc_grads = disc_tape.gradient(d_loss, model.discriminator.trainable_variables)
    model.disc_optimizer.apply_gradients(zip(disc_grads, model.discriminator.trainable_variables))

    return d_loss, adv_loss, content_loss, mse_loss

def train(model, dataset, log_iter, writer):
    
    Function that defines a single training step for the SR-GAN.
    Args:
        model: An object that contains tf keras compiled generator and
               discriminator models.
        dataset: A tf data object that contains low and high res images.
        log_iter: Number of iterations after which to add logs in
                  tensorboard.
        writer: Summary writer
    
    with writer.as_default():
        # Iterate over dataset
        for x, y in dataset:
            disc_loss, adv_loss, content_loss, mse_loss = train_step(model, x, y)
            # Log tensorboard summaries if log iteration is reached.
            if model.iterations % log_iter == 0:
                tf.summary.scalar('Adversarial Loss', adv_loss, step=model.iterations)
                tf.summary.scalar('Content Loss', content_loss, step=model.iterations)
                tf.summary.scalar('MSE Loss', mse_loss, step=model.iterations)
                tf.summary.scalar('Discriminator Loss', disc_loss, step=model.iterations)
                tf.summary.image('Low Res', tf.cast(255 * x, tf.uint8), step=model.iterations)
                tf.summary.image('High Res', tf.cast(255 * (y + 1.0) / 2.0, tf.uint8), step=model.iterations)
                tf.summary.image('Generated', tf.cast(255 * (model.generator.predict(x) + 1.0) / 2.0, tf.uint8),
                                 step=model.iterations)
                model.generator.save('models/generator.h5')
                model.discriminator.save('models/discriminator.h5')
                writer.flush()
            model.iterations += 1

epochs = 30000


# Initialize the GAN object.
gan = FastSRGAN(dataset_id)

# Run pre-training.
pretrain_generator(gan, ds, pretrain_summary_writer)

# Define the directory for saving the SRGAN training tensorbaord summary.
train_summary_writer = tf.summary.create_file_writer('logs/train')

# Run training.
for _ in range(epochs):
    train(gan, ds, 200, train_summary_writer)
"""
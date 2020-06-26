from gan import GAN

dataset_id="Irish"

gan = GAN(dataset_id)
gan.train(dataset_id,epochs=601, batch_size=32, save_interval=200)

from gan import GAN
import handshape_datasets as hd

dataset_id="Ciarp"

gan = GAN()


gan.train(dataset_id,epochs=30000, batch_size=32, save_interval=200)
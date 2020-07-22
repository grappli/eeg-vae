import tensorflow as tf

# MLP
class VAELayers:
    def __init__(self, hidden_dim, latent_dim, input_dim):
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.input_dim = input_dim
    
        
class MLPVAELayers(VAELayers):
    
    def encoder(self):
        layers = [tf.keras.layers.Flatten(),
                  tf.keras.layers.Dense(units=self.hidden_dim, activation=tf.nn.relu),
                  tf.keras.layers.Dense(units=self.hidden_dim, activation=tf.nn.relu),
                  tf.keras.layers.Dense(self.latent_dim *3),]
        return layers

    def decoder(self):
        layers = [tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
                  tf.keras.layers.Dense(units=self.hidden_dim, activation=tf.nn.relu),
                  tf.keras.layers.Dense(units=self.hidden_dim, activation=tf.nn.relu),
                  tf.keras.layers.Dense(units=self.input_dim, activation=tf.nn.sigmoid)]
        return layers
    
class EEGNetVAELayers(VAELayers):
    
    def __init__(self, hidden_dim, latent_dim, input_dim, n_channels, pool_size):
        super().__init__(hidden_dim, latent_dim, input_dim)
        self.n_channels = n_channels
        self.pool_size = pool_size

    def encoder(self):
        layers = [tf.keras.layers.Reshape((self.n_channels,128,1), input_shape=(self.n_channels*128,)),
                  tf.keras.layers.Conv2D(self.hidden_dim, (1,3), activation=tf.nn.elu),
                  tf.keras.layers.Conv2D(self.hidden_dim, (self.n_channels,1), activation=tf.nn.elu),
                  tf.keras.layers.Conv2D(self.hidden_dim, (1,10), activation=tf.nn.elu),
                  tf.keras.layers.MaxPool2D((1,3),(1,3)),
                  tf.keras.layers.Conv2D(self.hidden_dim, (1,10), activation=tf.nn.elu),
                  tf.keras.layers.MaxPool2D((1,3),(1,3)),
                  tf.keras.layers.Conv2D(self.hidden_dim, (1,10), activation=tf.nn.elu),
                  tf.keras.layers.Conv2D(self.latent_dim * 3, (1,1)),
                  tf.keras.layers.Flatten()]
        return layers

    def decoder(self):
        layers = [tf.keras.layers.Reshape((1,1,self.latent_dim)),
                  tf.keras.layers.UpSampling2D((self.pool_size[0],self.pool_size[1])),
                  tf.keras.layers.Conv2DTranspose(self.hidden_dim, (1,10), activation=tf.nn.elu, padding="same"),
                  tf.keras.layers.UpSampling2D((1,1)),
                  tf.keras.layers.Conv2DTranspose(self.hidden_dim, (1,10), activation=tf.nn.elu, padding="same"),
                  tf.keras.layers.Conv2DTranspose(self.hidden_dim, (1,10), activation=tf.nn.elu, padding="same"),
                  tf.keras.layers.Conv2DTranspose(self.hidden_dim, (self.n_channels,1), activation=tf.nn.elu, padding="same"),
                  tf.keras.layers.Conv2DTranspose(self.hidden_dim, (1,3), activation=tf.nn.elu, padding="same"),
                  tf.keras.layers.Conv2DTranspose(self.input_dim,(1,1), activation=tf.nn.sigmoid),
                  tf.keras.layers.Flatten()]
        return layers

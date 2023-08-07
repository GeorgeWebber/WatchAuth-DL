import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
import tensorflow_ranking as tfr
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, LSTM, Permute, Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, Reshape, SpatialDropout1D, Dropout, TimeDistributed
import tensorflow_probability as tfp




from keras.regularizers import l2, l1


from sklearn.decomposition import PCA
from sklearn.utils import class_weight, shuffle
from sklearn.metrics import f1_score
import sklearn

import pickle

from featurize import featurize

from scaler import CustomScaler



device_name = tf.test.gpu_device_name()


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    
class VAE_stats(keras.Model):
    def __init__(self, latent_dim, alpha = 0.1, beta = 0.001, gamma=0.1, delta=0.1, **kwargs):
        super(VAE_stats, self).__init__(**kwargs)
        self.encoder = get_encoder(latent_dim)
        self.decoder = get_decoder(latent_dim)
        self.auth = get_auth_model_from_latent_space(latent_dim)
        self.auth_out = get_auth_model_from_output(54)
        
        self.auth_on = True
        self.scaler = None
        

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.feature_loss_tracker = keras.metrics.Mean(name="feature_loss")
        self.auth_loss_tracker = keras.metrics.Mean(name="auth_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.stat_loss_tracker = keras.metrics.Mean(name="stat_loss")
        
        self.beta = beta
        self.alpha= alpha
        self.gamma = gamma
        self.delta = delta

    def save_model(self, folder, tag):
        
        with open(f'data/models/{folder}/enc_{tag}.pickle', 'wb') as f:
            pickle.dump(self.encoder, f)

        with open(f'data/models/{folder}/dec_{tag}.pickle', 'wb') as f:
            pickle.dump(self.decoder, f)

        with open(f'data/models/{folder}/auth_{tag}.pickle', 'wb') as f:
            pickle.dump(self.auth, f)
            
        with open(f'data/models/{folder}/auth2_{tag}.pickle', 'wb') as f:
            pickle.dump(self.auth_out, f)
        
        with open(f'data/models/{folder}/scaler_{tag}.pickle', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(f'data/models/{folder}/params_{tag}.pickle', 'wb') as f:
            params = {"feature_means":self.feature_means,
                      "feature_stds": self.feature_stds}
            pickle.dump(params, f)
    
    def load_model(self, folder, tag):
        
        with tf.keras.utils.custom_object_scope({'Sampling': Sampling, "SplitLayer":SplitLayer}):
            with open(f'data/models/{folder}/enc_{tag}.pickle', 'rb') as pickle_file:
                self.encoder = pickle.load(pickle_file)
                
            with open(f'data/models/{folder}/dec_{tag}.pickle', 'rb') as pickle_file:
                self.decoder = pickle.load(pickle_file)
                
            with open(f'data/models/{folder}/auth_{tag}.pickle', 'rb') as pickle_file:
                self.auth = pickle.load(pickle_file)
            
            with open(f'data/models/{folder}/auth2_{tag}.pickle', 'rb') as pickle_file:
                self.auth_out = pickle.load(pickle_file)
                
            with open(f'data/models/{folder}/scaler_{tag}.pickle', 'rb') as pickle_file:
                self.scaler = pickle.load(pickle_file)
            
            with open(f'data/models/{folder}/params_{tag}.pickle', 'rb') as pickle_file:
                params = pickle.load(pickle_file)
                self.feature_means = params["feature_means"]
                self.feature_stds = params["feature_stds"]
    
    def fit_scaler(self,data):
        self.scaler = CustomScaler()
        self.scaler.CHANNELS = 6
        self.scaler.fit(data)
        
        features = self.vae_featurize(data)
        self.feature_means = tf.cast(tf.math.reduce_mean(features, axis=0), tf.float32)
        self.feature_stds = tf.cast(tf.math.reduce_std(features, axis=0), tf.float32)
        
        print(self.feature_means.shape)
        
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.feature_loss_tracker,
            self.auth_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        data, y_user_one_hot = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            
            #reconstruction = self.rolling_mean(reconstruction)

            reconstruction_loss = 1/4 * (tf.reduce_mean(keras.losses.mse(data, reconstruction), axis=1) +
                                         self.lb_keogh(data, reconstruction, 16) +
                                         self.lb_keogh(data, reconstruction, 8) +
                                         self.lb_keogh(data, reconstruction, 4))
            
            #reconstruction = tfa.image.median_filter2d(reconstruction,filter_shape=(1,5), padding="constant")
            features = (self.vae_featurize(reconstruction) - self.feature_means) / self.feature_stds
            data_features = (self.vae_featurize(data) - self.feature_means) / self.feature_stds
            
            stat_loss = tf.reduce_mean(keras.losses.mean_absolute_error(data_features, features), axis=0)
            
            features_out = self.auth_out(features)
            feature_loss = tf.keras.losses.CategoricalCrossentropy()(y_user_one_hot, features_out)
            
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            
            auth_output_from_ls = self.auth(z)
            auth_loss = tfr.keras.losses.ApproxMRRLoss()(y_user_one_hot, auth_output_from_ls)
            
            
            total_loss = reconstruction_loss + self.beta * kl_loss + self.gamma * stat_loss
            if self.auth_on:
                total_loss += self.alpha * auth_loss
                total_loss += self.delta * feature_loss


        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.auth_loss_tracker.update_state(auth_loss)
        self.feature_loss_tracker.update_state(feature_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.stat_loss_tracker.update_state(stat_loss)
        return {
            "loss"               : self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "feature_loss"       : self.feature_loss_tracker.result(),
            "auth_loss"          : self.auth_loss_tracker.result(),
            "kl_loss"            : self.kl_loss_tracker.result(),
            "stat_loss"          : self.stat_loss_tracker.result()
        }

    def vae_featurize(self, data):
        
        _max = feature_max(data)
        _min = feature_min(data)
        _mean = feature_mean(data)
        _std = feature_stdev(data)
        _var = feature_var(data)
        _median = feature_median(data)
        _iqr = feature_iqr(data)
        _kurt = feature_kurt(data)
        _skew = feature_skew(data)
        
        features = tf.concat([_max, _min, _mean, _std, _var, _median, _iqr, _kurt, _skew], axis=1)
        return features


    def lb_keogh(self, ts1, ts2, w=10):

        u = self.rolling_max(ts1, w=w)
        l = self.rolling_min(ts1, w=w)

        above = ts2 > u
        below = ts2 < l

        above_sum = tf.reduce_sum((ts2[above]-u[above])**2) / 200 / 16
        below_sum = tf.reduce_sum((ts2[below]-l[below])**2) / 200 / 16

        lb_keogh_sum = above_sum + below_sum
        lb_keogh_sum = lb_keogh_sum / 128
        return lb_keogh_sum

    def median_filter(self, x, w=5):
        patches = tf.extract_image_patches(x, [1, k, k, 1], [1, k, k, 1], 4*[1], 'VALID')
        medians = tf.contrib.distributions.percentile(patches, 50, axis=3)

    def rolling_max(self, x,w=5):
        return tf.nn.pool(
            x,
            (w,),
            "MAX",
            strides=(1,),
            padding='SAME'
        )

    def rolling_mean(self, x,w=5):
        return tf.nn.max_pool1d(x, w, 1, padding='SAME')
    
    def rolling_min(self, x,w=5):
        return -tf.nn.pool(
            -x,
            (w,),
            "MAX",
            strides=(1,),
            padding='SAME'
        )

def feature_max(g_data):
	return tf.math.reduce_max(g_data, axis=1)

def feature_min(g_data):
	return tf.math.reduce_min(g_data, axis=1)
    
def feature_mean(g_data):
	return tf.math.reduce_mean(g_data, axis=1)
    
def feature_stdev(g_data):
	return tf.math.reduce_std(g_data, axis=1)

def feature_var(g_data):
	return tf.math.reduce_variance(g_data, axis=1)

def feature_median(g_data):
    return tfp.stats.percentile(g_data, 50.0, interpolation='midpoint', axis=1)

def feature_iqr(g_data):
    return tfp.stats.percentile(g_data, 75.0, interpolation='midpoint', axis=1) - tfp.stats.percentile(g_data, 25.0, interpolation='midpoint', axis=1)

def feature_kurt(g_data):
    std = tf.math.reduce_std(g_data, axis=1)
    diff = tf.math.reduce_mean((g_data - tf.math.reduce_mean(g_data, axis=1, keepdims = True))**4, axis=1)
    return diff / std**4

def feature_skew(g_data):
    std = tf.math.reduce_std(g_data, axis=1)
    diff = tf.math.reduce_mean((g_data - tf.math.reduce_mean(g_data, axis=1, keepdims = True))**3, axis=1)
    return diff / std**3



    
class SplitLayer(tf.keras.layers.Layer):
    def __init__(self, layers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = layers

    def call(self, inputs):
        return tf.gather(inputs,indices=self.layers,axis=-1)


def get_encoder(latent_dim):
    input_dim = (200,6)
    
    inputs = keras.Input(shape=input_dim)
    x = inputs

    xs = []
    
    for i in range(6):
        
        x = SplitLayer(i)(inputs)
        reshaped = Reshape((200, 1))(x)
        
        x = Conv1D(100, 10, strides=2, padding="same")(reshaped)
        x = MaxPooling1D(pool_size=2, strides=None, padding="same")(x)
        #x = SpatialDropout1D(0.2)(x)
        
        x = Conv1D(100, 3, strides=1, padding="same")(x)
        x = MaxPooling1D(pool_size=2, strides=None, padding="same")(x)
        #x = SpatialDropout1D(0.2)(x)

        xs.append(x)
    
    x = layers.Concatenate()(xs)

    
    x = LSTM(50)(x)

    z_mean = layers.Dense(latent_dim, name="z_mean" )(x) # kernel_initializer=keras.initializers.Zeros()
    z_log_var = layers.Dense(latent_dim, name="z_log_var", kernel_initializer=keras.initializers.Zeros())(x)

    z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")
    
    return encoder

def get_decoder(latent_dim):
    
    # assumes size of the output

    latent_inputs = keras.Input(shape=(latent_dim,))
    x = latent_inputs
    #x = layers.Dropout(rate=0.2)(x)
    #x = layers.Reshape((latent_dim))(x)
    x = layers.RepeatVector(25)(x)
    #x = layers.Permute((2,1))(x)
    x = layers.LSTM(100, activation="relu", input_shape=(25,latent_dim), return_sequences=True)(x)

    x = layers.TimeDistributed(layers.Dense(100, activation="relu"))(x)

    x = layers.Conv1D(100,4, padding="same")(x)
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(100,4, padding="same")(x)
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(50,8, padding="same")(x)
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(6,8, padding="same")(x)

    decoder_outputs = x

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    return decoder

def get_auth_model_from_latent_space(latent_dim):
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = latent_inputs
    
    x = tf.split(x, 5, axis=1)[0]

    x = layers.Dense(16, activation="softmax")(x)
    #x = layers.Softmax()(x)

    auth_outputs = x

    auth = keras.Model(latent_inputs, auth_outputs, name="auth_from_latent_space")
    return auth

def get_auth_model_from_output(features):
    inputs = keras.Input(shape=(features,))
    x = inputs
    x = layers.Dense(25, activation="relu")(x)
    x = layers.Dropout(rate=0.2)(x)
    x = layers.Dense(25, activation="relu")(x)
    x = layers.Dropout(rate=0.2)(x)
    x = layers.Dense(16, activation="softmax")(x)

    auth_outputs = x

    auth = keras.Model(inputs, auth_outputs, name="auth_from_output")
    return auth
    

def get_discriminator():
    input_dim = (200,16)
    inputs = keras.Input(shape=input_dim)
    x = inputs
    
    xs = []
    
    for i in range(6):
        
        x = SplitLayer(i)(inputs)
        reshaped = Reshape((200, 1))(x)
        
        x = Conv1D(100, 10, strides=2, padding="same")(reshaped)   #, kernel_regularizer=l2(1e-5)
        x = MaxPooling1D(pool_size=2, strides=None, padding="same")(x)
        
        x = Conv1D(100, 3, strides=1,  padding="same")(x)     #   kernel_regularizer=l2(1e-5),
        x = MaxPooling1D(pool_size=2, strides=None, padding="same")(x)

        
        
        xs.append(x)
    
    x = layers.Concatenate()(xs)
    x = LSTM(50)(x)

    x = Dense(25, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(10, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation="sigmoid")(x)
    #x = layers.Lambda(lambda x: x / 2)
    out = x

    model = keras.Model(inputs, out, name="differentiating_fake_gestures")
    
    return model

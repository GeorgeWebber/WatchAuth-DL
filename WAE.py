import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
import tensorflow_ranking as tfr
#import tensorflow_addons as tfa
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

from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

    
    
class WAE(keras.Model):
    def __init__(self, encoder, decoder, _classifier = None, _auth = None, alpha = 0.1, beta = 0.1, gamma = 1, scale_included=True, use_stats=False, **kwargs):
        super(WAE, self).__init__(**kwargs)

        self.encoder = encoder
        self.decoder = decoder
        self.classifier = _classifier
        self.auth = _auth
        
        self.loss_func = "LB_KEOGH_mod"
        self.auth_on = True
        self.wae_loss_on = True
        self.scale_included = scale_included
        self.scaler = None
        self.use_stats = use_stats
        
        self.loss_trackers = {}
        for loss_name in ["total_loss", "reconstruction_loss", "wae_loss", "classification_loss", "feature_loss", "reencoding_loss", "auth_loss"]:
            self.loss_trackers[loss_name] = keras.metrics.Mean(name=loss_name)
        
        self.beta = beta
        self.alpha= alpha
        self.gamma = gamma

    def save_model(self, folder, tag):
        with suppress_stdout():
            with open(f'data/models/{folder}/enc_{tag}.pickle', 'wb') as f:
                # Pickle the 'data' dictionary using the highest protocol available.
                pickle.dump(self.encoder, f)

            with open(f'data/models/{folder}/dec_{tag}.pickle', 'wb') as f:
                # Pickle the 'data' dictionary using the highest protocol available.
                pickle.dump(self.decoder, f)

            with open(f'data/models/{folder}/auth_{tag}.pickle', 'wb') as f:
                # Pickle the 'data' dictionary using the highest protocol available.
                pickle.dump(self.auth, f)

            if self.scale_included:
                with open(f'data/models/{folder}/scaler_{tag}.pickle', 'wb') as f:
                    # Pickle the 'data' dictionary using the highest protocol available.
                    pickle.dump(self.scaler, f)
            if self.use_stats:
                with open(f'data/models/{folder}/params_{tag}.pickle', 'wb') as f:
                    params = {"feature_means":self.feature_means,
                              "feature_stds": self.feature_stds}
                    pickle.dump(params, f)
    
    def load_model(self, folder, tag):
        
        with suppress_stdout():
            with tf.keras.utils.custom_object_scope({'Sampling': Sampling, "SplitLayer":SplitLayer}):
                with open(f'data/models/{folder}/enc_{tag}.pickle', 'rb') as pickle_file:
                    self.encoder = pickle.load(pickle_file)
                with open(f'data/models/{folder}/dec_{tag}.pickle', 'rb') as pickle_file:
                    self.decoder = pickle.load(pickle_file)
                with open(f'data/models/{folder}/auth_{tag}.pickle', 'rb') as pickle_file:
                    self.auth = pickle.load(pickle_file)
                if self.scale_included:
                    with open(f'data/models/{folder}/scaler_{tag}.pickle', 'rb') as pickle_file:
                        # Pickle the 'data' dictionary using the highest protocol available.
                        self.scaler = pickle.load(pickle_file)
                
                if self.use_stats:
                    with open(f'data/models/{folder}/params_{tag}.pickle', 'rb') as pickle_file:
                        params = pickle.load(pickle_file)
                        self.feature_means = params["feature_means"]
                        self.feature_stds = params["feature_stds"]
    
    def fit_scaler(self,data):
        self.scale_included = True
        self.scaler = CustomScaler()
        self.scaler.CHANNELS=6
        self.scaler.fit(data)
        
        if self.use_stats:
            features = self.vae_featurize(data)
            self.feature_means = tf.cast(tf.math.reduce_mean(features, axis=0), tf.float32)
            self.feature_stds = tf.cast(tf.math.reduce_std(features, axis=0), tf.float32)
    
    @property
    def metrics(self):
        return self.loss_trackers.values()

    def wasserstein_loss(self, z):
        batch_size = tf.shape(z)[0]
        dim = tf.shape(z)[1]

        zis = tf.keras.backend.random_normal(shape=(batch_size, dim))
    
        A = tf.concat([z,zis], 0)
        r = tf.reduce_sum(A*A, 1)

        r = tf.reshape(r, [-1, 1])
        D = r - 2*tf.matmul(A, tf.transpose(A)) + tf.transpose(r)
        D = tf.math.sqrt(tf.clip_by_value(D, clip_value_min=1e-8, clip_value_max=1e8))

        zs_pairwise = tf.reduce_sum(D[:batch_size, :batch_size])
        zis_pairwise = tf.reduce_sum(D[batch_size:, batch_size:])
        combo_pairwise = tf.reduce_sum(D[:batch_size, batch_size:])

        batch = tf.cast(batch_size, dtype=tf.float32)

        #kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        wae_loss = ( 0.5/((batch-1)*(batch)) * (zs_pairwise + zis_pairwise)
                    - 2/((batch)*(batch)) * combo_pairwise  )
        return wae_loss
    
    def get_losses(self, data, reconstruction, z, y_user_one_hot, training=True):
        
        
        if self.loss_func == "MSE":
            reconstruction_loss = tf.reduce_mean(keras.losses.mse(data, reconstruction)) # , axis=1
        elif self.loss_func == "LB_KEOGH":
            reconstruction_loss = self.lb_keogh(data, reconstruction, 10)
        elif self.loss_func == "LB_KEOGH_mod":
            reconstruction_loss = 1/15 * (
                                     1*self.lb_keogh(data, reconstruction, 16) + 
                                     2*self.lb_keogh(data, reconstruction, 8) + 
                                     3*self.lb_keogh(data, reconstruction, 4) + 
                                     4*self.lb_keogh(data, reconstruction, 2) + 
                                     5*self.lb_keogh(data, reconstruction, 1))
            """
            reconstruction_loss = 1/7 * (self.lb_keogh(data, reconstruction, 128) + 
                                     self.lb_keogh(data, reconstruction, 64) + 
                                     self.lb_keogh(data, reconstruction, 32) + 
                                     self.lb_keogh(data, reconstruction, 16) + 
                                     self.lb_keogh(data, reconstruction, 8) + 
                                     self.lb_keogh(data, reconstruction, 4) + 
                                     self.lb_keogh(data, reconstruction, 2) + 
                                     self.lb_keogh(data, reconstruction, 1))
            """
        else:
            assert False
        
        if self.wae_loss_on:
            wae_loss = self.wasserstein_loss(z)
        else:
            wae_loss = reconstruction_loss - reconstruction_loss
        
        if self.use_stats:
            feature_loss = 0
            features = (self.vae_featurize(reconstruction) - self.feature_means) / self.feature_stds
            data_features = (self.vae_featurize(data) - self.feature_means) / self.feature_stds

            feature_loss = tf.reduce_mean(keras.losses.mean_absolute_error(data_features, features), axis=0)
            """
            for i in range(9):
                features = (self.vae_featurize(tf.slice(reconstruction, [0, 20*i, 0], [-1, 40, -1])) - self.feature_means) / self.feature_stds
                data_features = (self.vae_featurize(tf.slice(data, [0, 20*i, 0], [-1, 40, -1])) - self.feature_means) / self.feature_stds
                feature_loss += tf.reduce_mean(keras.losses.mean_absolute_error(data_features, features), axis=0) / 9
            """
        else:
            feature_loss = wae_loss - wae_loss # i.e. zero
        
        if self.auth_on:
            auth_output_from_ls = self.auth(z, training=training)
            auth_loss = tfr.keras.losses.ApproxMRRLoss()(y_user_one_hot, auth_output_from_ls)
        else:
            auth_loss = wae_loss - wae_loss # i.e. zero
        
        loss = reconstruction_loss + self.beta * wae_loss + self.gamma * feature_loss + self.alpha * auth_loss
        
        loss_dict = {"total_loss" : loss, "reconstruction_loss" : reconstruction_loss, "wae_loss": wae_loss, "feature_loss": feature_loss, "auth_loss":auth_loss}
        return loss_dict
    
    def train_step(self, data):
        data, y_user_one_hot = data
        with tf.GradientTape() as tape:
            z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            loss_dict = self.get_losses(data, reconstruction, z, y_user_one_hot, training=True)
            
            
            

            """
            if self.auth_on:
                total_loss = self.alpha * auth_loss + sample_loss + self.beta * kl_loss #+ classification_loss # + 0.1*feature_loss
            else:
                total_loss = sample_loss + self.beta * kl_loss
            """

        grads = tape.gradient(loss_dict["total_loss"], self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        for key in loss_dict:
            self.loss_trackers[key].update_state(loss_dict[key])
        return {k:v.result() for (k,v) in self.loss_trackers.items()}


    def test_step(self, data):
        data, y_user_one_hot = data
        
        z = self.encoder(data, training=False)
        reconstruction = self.decoder(z, training=False)
        
        losses = self.get_losses(data, reconstruction, z, y_user_one_hot, training=False) # , training=False?
        return losses
    
    """
    def train_step(self, data):
        data, y_user_one_hot = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            z_mean_reconstruction = self.decoder(z_mean)
            z_mean_features = self.vae_featurize(z_mean_reconstruction)
            true_features = self.vae_featurize(data)
            
            feature_loss = tf.reduce_mean(keras.losses.mean_absolute_error)
            
            auth_output = self.auth(z)
            auth_loss = tf.keras.losses.CategoricalCrossentropy()(y_user_one_hot, auth_output)
            
            batch = tf.shape(data)[0]
            dim = tf.shape(z_mean)[1]
            
            zis = tf.keras.backend.random_normal(shape=(batch, dim))

            reconstruction_loss = 1/4 * (tf.reduce_mean(keras.losses.mse(data, reconstruction), axis=1) +
                                         self.lb_keogh(data, reconstruction, 16) +
                                         self.lb_keogh(data, reconstruction, 8) +
                                         self.lb_keogh(data, reconstruction, 4))
            A = tf.concat([z,zis], 0)
            r = tf.reduce_sum(A*A, 1)

            r = tf.reshape(r, [-1, 1])
            D = r - 2*tf.matmul(A, tf.transpose(A)) + tf.transpose(r)
            D = tf.math.sqrt(tf.clip_by_value(D, clip_value_min=1e-8, clip_value_max=1e8))
            
            zs_pairwise = tf.reduce_sum(D[:batch, :batch])
            zis_pairwise = tf.reduce_sum(D[batch:, batch:])
            combo_pairwise = tf.reduce_sum(D[:batch, batch:])
            
            batch = tf.cast(batch, dtype=tf.float32)
            
            #kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            wae_loss = ( 0.5/((batch-1)*(batch)) * (zs_pairwise + zis_pairwise)
                        - 2/((batch)*(batch)) * combo_pairwise  )

            total_loss = reconstruction_loss + self.beta * wae_loss + self.alpha*auth_loss
            
            


        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.auth_loss_tracker.update_state(auth_loss)
        #self.feature_loss_tracker.update_state(feature_loss)
        self.kl_loss_tracker.update_state(wae_loss)
        return {
            "loss"               : self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            #"feature_loss"       : self.feature_loss_tracker.result(),
            "auth_loss"          : self.auth_loss_tracker.result(),
            "wae_loss"            : self.kl_loss_tracker.result(),
        }
    """

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

        batch_size = tf.shape(ts1)[0]
        batch_size = tf.cast(batch_size, dtype=tf.float32)
        
        u = self.rolling_max(ts1, w=w)
        l = self.rolling_min(ts1, w=w)

        above = ts2 > u
        below = ts2 < l

        above_sum = tf.reduce_sum((ts2[above]-u[above])**2) / 200 / 16
        below_sum = tf.reduce_sum((ts2[below]-l[below])**2) / 200 / 16

        lb_keogh_sum = above_sum + below_sum
        lb_keogh_sum = lb_keogh_sum / batch_size
        return lb_keogh_sum

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



class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


    
class SplitLayer(tf.keras.layers.Layer):
    def __init__(self, layers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = layers

    def call(self, inputs):
        return tf.gather(inputs,indices=self.layers,axis=-1)


def get_encoder(latent_dim=10):
    input_dim=(200,6)
    inputs = keras.Input(shape=input_dim)
    x = inputs
    y = inputs
    
    last_output_channels = 22
    for a in range(4):
        x0 = Conv1D(last_output_channels, 1, strides=1, padding="same", activation="relu")(x)
        x1 = Conv1D(last_output_channels, 3, strides=1, padding="same", activation="relu")(x)
        x2 = Conv1D(last_output_channels, 5, strides=1, padding="same", activation="relu")(x)
        x3 = Conv1D(last_output_channels, 10, strides=1, padding="same", activation="relu")(x)
        x0 = MaxPooling1D(pool_size=2, strides=None, padding="same")(x0)
        x1 = MaxPooling1D(pool_size=2, strides=None, padding="same")(x1)
        x2 = MaxPooling1D(pool_size=2, strides=None, padding="same")(x2)
        x3 = MaxPooling1D(pool_size=2, strides=None, padding="same")(x3)
        x = layers.Concatenate()([x0,x1,x2,x3])
        
        x = layers.TimeDistributed(Dense(30, activation="relu"))(x) # , kernel_regularizer=keras.regularizers.L1(1e-4)))(x) #  
        
    x = layers.GRU(32, return_sequences=True)(x)
    x = layers.GRU(32, return_sequences=True)(x)
    x = layers.GRU(max(32, latent_dim))(x)
    
    out = layers.Dense(latent_dim)(x)
    
    encoder = keras.Model(inputs, out, name="encoder")
    return encoder

def get_decoder(latent_dim=10):

    latent_inputs = keras.Input(shape=(latent_dim,))
    x = latent_inputs
    x = layers.RepeatVector(25)(x)
    x = layers.GRU(128, return_sequences=True)(x)
    x = layers.GRU(128, return_sequences=True)(x)
    x = layers.GRU(128, return_sequences=True)(x)

    x = layers.TimeDistributed(Dense(64, activation="relu"))(x)

    x = layers.Conv1D(64,3, padding="same", activation="relu")(x)
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(32,3, padding="same", activation="relu")(x)
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(16,3, padding="same", activation="relu")(x)
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(6,3, padding="same")(x)

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
    x = layers.Dense(25, activation="relu")(x)
    x = layers.Dense(16, activation="softmax")(x)

    auth_outputs = x

    auth = keras.Model(inputs, auth_outputs, name="auth_from_output")
    return auth
    

def get_discriminator():
    input_dim = (200,16)
    inputs = keras.Input(shape=input_dim)
    x = inputs
    
    xs = []
    
    for i in range(16):
        
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

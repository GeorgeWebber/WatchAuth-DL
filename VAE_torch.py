import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.decomposition import PCA
from sklearn.utils import class_weight
from sklearn.metrics import f1_score
import sklearn

import tensorflow as tf


import pickle

from featurize import featurize

from scaler import CustomScaler


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

import torch.nn as nn
import torch.nn.functional as F

class Sampling(nn.Module):
    """Uses (z_mean, z_log_var) to sample z."""

    def forward(self, inputs):
        z_mean, z_log_var = inputs
        batch = z_mean.size(0)
        dim = z_mean.size(1)
        epsilon = torch.randn(batch, dim, device=z_mean.device)
        return z_mean #+ torch.exp(0.5 * z_log_var) * epsilon

class Encoder(nn.Module):
    """ (batch, channels, time) """
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        
        self.latent_dim = latent_dim
        
        self.max_pool = nn.MaxPool1d(kernel_size=2)
        self.convs = []
        self.time_distributed = []
        input_sizes = [6,30,30,30]
        self.conv11 = nn.Conv1d(6, 22, kernel_size=1, stride=1, padding="same")
        self.conv12 = nn.Conv1d(6, 22, kernel_size=3, stride=1, padding="same")
        self.conv13 = nn.Conv1d(6, 22, kernel_size=5, stride=1, padding="same")
        self.conv14 = nn.Conv1d(6, 22, kernel_size=10, stride=1, padding="same")
        
        self.conv21 = nn.Conv1d(30, 22, kernel_size=1, stride=1, padding="same")
        self.conv22 = nn.Conv1d(30, 22, kernel_size=3, stride=1, padding="same")
        self.conv23 = nn.Conv1d(30, 22, kernel_size=5, stride=1, padding="same")
        self.conv24 = nn.Conv1d(30, 22, kernel_size=10, stride=1, padding="same")
        
        self.conv31 = nn.Conv1d(30, 22, kernel_size=1, stride=1, padding="same")
        self.conv32 = nn.Conv1d(30, 22, kernel_size=3, stride=1, padding="same")
        self.conv33 = nn.Conv1d(30, 22, kernel_size=5, stride=1, padding="same")
        self.conv34 = nn.Conv1d(30, 22, kernel_size=10, stride=1, padding="same")
        
        self.conv41 = nn.Conv1d(30, 22, kernel_size=1, stride=1, padding="same")
        self.conv42 = nn.Conv1d(30, 22, kernel_size=3, stride=1, padding="same")
        self.conv43 = nn.Conv1d(30, 22, kernel_size=5, stride=1, padding="same")
        self.conv44 = nn.Conv1d(30, 22, kernel_size=10, stride=1, padding="same")
        
        for a in range(4):
            self.time_distributed.append(nn.Conv2d(1, 30, (88,1)))
        
        self.time_distributed = nn.ModuleList(self.time_distributed)
            
        self.GRU1 = nn.GRU(input_size=30, hidden_size=32, batch_first=True)
        self.GRU2 = nn.GRU(input_size=32, hidden_size=32, batch_first=True)
        self.GRU3 = nn.GRU(input_size=32, hidden_size=max(32, latent_dim), batch_first=True)
        self.dense_mean = nn.Linear(max(32, latent_dim), self.latent_dim)
        self.dense_log_var = nn.Linear(max(32, latent_dim), self.latent_dim)

    def forward(self, inputs):
        
        x = inputs
        
        x1 = nn.MaxPool1d(kernel_size=2)(F.relu(self.conv11(x)))
        x2 = nn.MaxPool1d(kernel_size=2)(F.relu(self.conv12(x)))
        x3 = nn.MaxPool1d(kernel_size=2)(F.relu(self.conv13(x)))
        x4 = nn.MaxPool1d(kernel_size=2)(F.relu(self.conv14(x)))
        x = torch.concat([x1,x2,x3,x4], axis=1)

        x = torch.unsqueeze(x, 1)
        x = self.time_distributed[0](x)
        x = torch.squeeze(x)
        x = F.relu(x)
    
        x1 = nn.MaxPool1d(kernel_size=2)(F.relu(self.conv21(x)))
        x2 = nn.MaxPool1d(kernel_size=2)(F.relu(self.conv22(x)))
        x3 = nn.MaxPool1d(kernel_size=2)(F.relu(self.conv23(x)))
        x4 = nn.MaxPool1d(kernel_size=2)(F.relu(self.conv24(x)))
        x = torch.concat([x1,x2,x3,x4], axis=1)
        
        x = torch.unsqueeze(x, 1)
        x = self.time_distributed[1](x)
        x = torch.squeeze(x)
        x = F.relu(x)
        
        x1 = nn.MaxPool1d(kernel_size=2)(F.relu(self.conv31(x)))
        x2 = nn.MaxPool1d(kernel_size=2)(F.relu(self.conv32(x)))
        x3 = nn.MaxPool1d(kernel_size=2)(F.relu(self.conv33(x)))
        x4 = nn.MaxPool1d(kernel_size=2)(F.relu(self.conv34(x)))
        x = torch.concat([x1,x2,x3,x4], axis=1)
        
        x = torch.unsqueeze(x, 1)
        x = self.time_distributed[2](x)
        x = torch.squeeze(x)
        x = F.relu(x)
                
        x1 = nn.MaxPool1d(kernel_size=2)(F.relu(self.conv41(x)))
        x2 = nn.MaxPool1d(kernel_size=2)(F.relu(self.conv42(x)))
        x3 = nn.MaxPool1d(kernel_size=2)(F.relu(self.conv43(x)))
        x4 = nn.MaxPool1d(kernel_size=2)(F.relu(self.conv44(x)))
        x = torch.concat([x1,x2,x3,x4], axis=1)
        
        x = torch.unsqueeze(x, 1)
        x = self.time_distributed[3](x)
        x = torch.squeeze(x)
        x = F.relu(x)
            
        x = torch.transpose(x, 1,2)
        x, _ = self.GRU1(x)
        x, _ = self.GRU2(x)
        _, x = self.GRU3(x)
        x = x[0]
        x = x.squeeze()
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = Sampling()([z_mean, z_log_var])
        return z_mean, z_log_var, z


"""
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
 """
    
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim

        self.repeat = nn.ReplicationPad1d((0, 0, 24, 0))
        self.GRU1 = nn.GRU(input_size=latent_dim,hidden_size=128,batch_first=True, num_layers =3)
        
        gru_out_dim = 128
        
        self.upsample = nn.Upsample(scale_factor=2, align_corners=False)
        
        self.tdd = nn.Conv2d(1, 64, (128, 1))
        
        self.out_conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(), nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(), nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(), nn.Upsample(scale_factor=2, mode="nearest"), 
            nn.Conv1d(in_channels=16, out_channels=6, kernel_size=3, stride=1, padding="same")
        )

    def forward(self, latent_inputs):
        
        x = latent_inputs
        x = self.repeat(x.unsqueeze(1)).squeeze(1)
        
        x, _ = self.GRU1(x)
        
        x = torch.transpose(x, 1,2)
        
        x = torch.unsqueeze(x, 1)
        x = self.tdd(x)
        x = torch.squeeze(x)
        x = F.relu(x)
    
        
        x = self.out_conv_layers(x)
        decoder_outputs = x

        return decoder_outputs
    

    
class VAE(nn.Module):
    def __init__(self, latent_dim, beta=0.001, alpha=0.1):
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()
        
        self.beta = beta
        self.alpha = alpha
    
    def get_encoder(self):
        encoder = Encoder(self.latent_dim)
        return encoder
    
    def get_decoder(self):
        decoder = Decoder(self.latent_dim)
        return decoder
    

    def fit_scaler(self,data):
        self.scale_included = True
        self.scaler = CustomScaler()
        self.scaler.CHANNELS=6
        self.scaler.fit(data)
    
    def save_model(self, folder, tag):
        #with suppress_stdout():
        with open(f'data/models/{folder}/enc_{tag}.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(self.encoder, f)

        with open(f'data/models/{folder}/dec_{tag}.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(self.decoder, f)

        if self.scale_included:
            with open(f'data/models/{folder}/scaler_{tag}.pickle', 'wb') as f:
                # Pickle the 'data' dictionary using the highest protocol available.
                pickle.dump(self.scaler, f)
            
    def load_model(self, folder, tag):
        with tf.keras.utils.custom_object_scope({'Sampling': Sampling}):
            with open(f'data/models/{folder}/enc_{tag}.pickle', 'rb') as pickle_file:
                self.encoder = pickle.load(pickle_file)
            with open(f'data/models/{folder}/dec_{tag}.pickle', 'rb') as pickle_file:
                self.decoder = pickle.load(pickle_file)
            with open(f'data/models/{folder}/scaler_{tag}.pickle', 'rb') as pickle_file:
                self.scaler = pickle.load(pickle_file)
    
"""  
class VAE(keras.Model):
    def __init__(self, encoder, decoder, _classifier = None, _auth = None, beta = 0.001, alpha = 0.1, scale_included=True, loss_func = "MSE",**kwargs):



def get_auth_model(input_dim):
    inputs = keras.Input(shape=input_dim)
    x = inputs
    x = Reshape((200, 16, 1))(x)
    x = Conv2D(10,(3, 1), kernel_regularizer=l2(0.01), strides=(1,1), padding="same")(x)
    x = MaxPooling2D(pool_size=(2,1), strides=None, padding="same")(x)

    x = Conv2D(100,(2,16),strides=(1,1), kernel_regularizer=l1(0.01))(x)

    x = MaxPooling2D(pool_size=(2,1), strides=None, padding="same")(x)
    x = Permute((1,3,2))(x)
    x = Reshape((50,100))(x)

    x = layers.SpatialDropout1D(0.2)(x)
    x = Conv1D(50, 4, strides=2, padding="same")(x)
    x = MaxPooling1D(pool_size=2, strides=None, padding="same")(x)

    x = layers.SpatialDropout1D(0.2)(x)
    x = Conv1D(50, 4, strides=2, padding="same")(x)

    x = LSTM(50)(x)

    x = Dense(25, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(10, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(1, activation="sigmoid")(x)
    out = x

    model = keras.Model(inputs, out, name="LSTM_classifier")
    return model


def get_encoder(input_dim, latent_dim=200):

    inputs = keras.Input(shape=input_dim)

    x = inputs
    x = Reshape((200, 16, 1))(x)
    x = Conv2D(10, (3, 1), kernel_regularizer=l2(0.01), strides=(1,1), padding="same")(x)
    x = MaxPooling2D(pool_size=(2,1), strides=None, padding="same")(x)

    x = Conv2D(100,(2,16), kernel_regularizer=l1(0.01), strides=(1,1))(x)

    x = MaxPooling2D(pool_size=(2,1), strides=None, padding="same")(x)
    x = Permute((1,3,2))(x)
    x = Reshape((50,100))(x)

    x = layers.SpatialDropout1D(0.2)(x)
    x = Conv1D(50, 4, strides=2, padding="same")(x)
    x = MaxPooling1D(pool_size=2, strides=None, padding="same")(x)

    x = layers.SpatialDropout1D(0.2)(x)
    x = Conv1D(50, 4, strides=2, padding="same")(x)

    x = LSTM(latent_dim)(x)

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
    x = layers.TimeDistributed(layers.Dense(16))(x)

    x = layers.Conv1D(16,4, padding="same")(x)
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(16,4, padding="same")(x)
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(16,8, padding="same")(x)
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(16,8, padding="same")(x)

    x2 = layers.RepeatVector(25)(latent_inputs)

    decoder_outputs = x

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    return decoder


def get_auth_model_from_latent_space(latent_dim):
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = latent_inputs
    
    x = tf.split(x, 5, axis=1)[0]
    
    x = layers.Dense(25, activation="relu")(x)
    x = layers.Dense(16, activation="softmax")(x)
    #x = layers.Softmax()(x)

    auth_outputs = x

    auth = keras.Model(latent_inputs, auth_outputs, name="auth")
    return auth
    
    
"""


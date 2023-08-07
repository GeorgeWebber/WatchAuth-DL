import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
import tensorflow_ranking as tfr
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, LSTM, Permute, Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, Reshape, Dropout

from keras.regularizers import l2, l1
from keras.optimizers import Adam

import sklearn
from sklearn.decomposition import PCA
from sklearn.utils import class_weight
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import pickle

from VAE_experimental import VAE_GAN
from scaler import CustomScaler

def run(users=range(16)):

    device_name = tf.test.gpu_device_name()

    file_name = "raw_with_maps" # or offsets_2

    x_data = np.load(f"data/processed/x_{file_name}.npy")
    y_user = np.load(f"data/processed/y_user_{file_name}.npy")
    y_intent = np.load(f"data/processed/y_intent_{file_name}.npy").astype(int)
    y_gesture = np.load(f"data/processed/y_gesture_type_{file_name}.npy")

    train_gesture_map = np.load(f"data/processed/train_gesture_map_{file_name}.npy").astype(int)
    test_gesture_map = np.load(f"data/processed/test_gesture_map_{file_name}.npy").astype(int)

    ###
    latent_dim = 50
    
    real_gestures_allowed_per_terminal = 2

    for auth_user in users:
        print(f"Auth user is {auth_user}")
        
        vae = VAE_GAN(latent_dim)
        vae.fit_scaler(x_data[(y_intent==0)])
        vae.auth_on = False
        vae.gamma = 0
        
        
        initial_map = ((y_user.argmax(axis=1) != auth_user) & (train_gesture_map==1)) | (y_intent == 0)
        train_data = vae.scaler.transform(x_data[initial_map])
        
        print("Fitting basic")

        vae.beta = 1e-4
        vae.compile(Adam())
        history_1 = vae.fit(train_data, y_user[initial_map], epochs=10, batch_size=128, verbose=0)
        vae.save_model(f"no_{auth_user}", f"vae_gan_basic_reconstruction_{latent_dim}")

        

        real_gesture_indices = np.zeros(len(y_user))

        for terminal_type in [3,4,5,6,7,8,9]:
            legal_gesture_indices = (train_gesture_map==1) & (y_user.argmax(axis=1) == auth_user) & (y_gesture.argmax(axis=1) == terminal_type)

            legal_gesture_indices = np.nonzero(legal_gesture_indices)[0]
            legal_gesture_indices = legal_gesture_indices[:real_gestures_allowed_per_terminal]
            
            real_gesture_indices[legal_gesture_indices] = 1
        
        specific_map = ((y_user.argmax(axis=1) != auth_user) & (train_gesture_map==1)) | (real_gesture_indices==1)
        
        print(np.unique(real_gesture_indices, return_counts=True))
        
        beta = 1e-4
        vae.beta = beta
        vae.alpha = 0.01
        vae.gamma = 0.01
        vae.auth_on = True
        
        vae.compile(Adam())

        epochs = 60
        train_data = vae.scaler.transform(x_data[specific_map])

        for i in range(30):
            print(f"Fitting specific {i}")

            vae.discriminator.trainable = False
            vae.discriminator.compile(Adam())
            vae.compile(Adam())

            vae.fit(train_data, y_user[specific_map], epochs=epochs, batch_size=128, verbose=0)

            vae.train_discriminator(train_data, max_epochs=10, verbose=0)

            if (i+1) % 10 == 0: 
                vae.save_model(f"no_{auth_user}", f"vae_gan_specific_{latent_dim}_gestures={real_gestures_allowed_per_terminal}")

        vae.compile(Adam(1e-5))
        vae.fit(train_data, y_user[specific_map], epochs=200, batch_size=128, verbose=1)
                
        vae.save_model(f"no_{auth_user}", f"vae_gan_specific_{latent_dim}_gestures={real_gestures_allowed_per_terminal}")

        
       
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

from VAE import Sampling, VAE, get_auth_model_from_latent_space, get_decoder, get_encoder
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
    latent_dim = 100

    for auth_user in users:
        print(f"Auth user is {auth_user}")

        encoder = get_encoder(x_data.shape[1:], latent_dim)
        decoder = get_decoder(latent_dim)
        auth = get_auth_model_from_latent_space(latent_dim)

        vae = VAE(encoder, decoder, None, auth, scale_included=True)
        vae.fit_scaler(x_data[(y_intent==0)])
        vae.auth_on = False

        initial_map = ((y_user.argmax(axis=1) != auth_user) & (train_gesture_map==1)) | (y_intent == 0)
        train_data = vae.scaler.transform(x_data[initial_map])
        
        print("Fitting basic")

        vae.beta = 0.000001
        vae.compile(Adam())
        history_1 = vae.fit(train_data, y_user[initial_map], epochs=10, batch_size=128, verbose=0)
        vae.save_model(f"no_{auth_user}", f"basic_reconstruction_{latent_dim}")


        specific_map = (y_user.argmax(axis=1) != auth_user) & (train_gesture_map==1)  # change with more restricted data in future
        vae = VAE(None, None, None, None)
        vae.load_model(f"no_{auth_user}", f"basic_reconstruction_{latent_dim}")

        
        beta = 0.000001
        vae.beta = beta
        vae.auth_on = True

        vae.compile(Adam())

        epochs = 10
        train_data = vae.scaler.transform(x_data[specific_map])
        
        print("Fitting specific")

        vae.fit(train_data, y_user[specific_map], epochs=100, batch_size=128, verbose=0)

        for epoch in range(epochs):
            print(f"Epoch {epoch}")
            beta_star = beta / np.exp(-epoch * np.log(10000) / epochs)
            vae.beta = beta_star
            vae.compile(Adam())

            vae.fit(train_data, y_user[specific_map], epochs=100, batch_size=128, verbose=0)

        vae.save_model(f"no_{auth_user}", f"trained_without_user_big_beta")

        """

        real_gestures_allowed_per_terminal = 1
        gestures_to_generate = 500

        real_gesture_list = []

        for terminal_type in [3,4,5,6,7,8,9]:

            legal_gestures = x_data[(train_gesture_map==1) & (y_user.argmax(axis=1) == auth_user) & (y_gesture.argmax(axis=1) == terminal_type) ]

            real_gesture_list.append(legal_gestures[:real_gestures_allowed_per_terminal])

        real_gestures = np.concatenate(real_gesture_list)

        #real_gestures = np.repeat(real_gestures, 1 + gestures_to_generate // (7 * real_gestures_allowed_per_terminal), axis=0)[:gestures_to_generate]



        enc = vae.encoder.predict(vae.scaler.transform(real_gestures))[2]

        convex_hull_points = []


        for i in range(gestures_to_generate - (7*real_gestures_allowed_per_terminal)): 
            s = (np.random.pareto(3, 7))
            s = s / sum(s)

            new_point = np.dot(enc.transpose(), s)

            convex_hull_points.append([new_point])

        convex_hull_points = np.concatenate(convex_hull_points)

        enc = np.concatenate([enc, convex_hull_points])

        dec = vae.scaler.inverse_transform(vae.decoder.predict(enc))


        np.save(f"data/generated_samples/{auth_user}_limited_data_extra_data_self_mixed.npy", dec)
        """

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
import tensorflow_ranking as tfr
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, LSTM, Permute, Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, Reshape, Dropout

from keras.regularizers import l2, l1


from sklearn.decomposition import PCA
from sklearn.utils import class_weight
from sklearn.metrics import f1_score
import sklearn

import pickle
from featurize import featurize

from sklearn.manifold import TSNE
import pandas as pd


import random

from scipy.stats import multivariate_normal


def dimension_reduction(z_pca, labels) -> pd.DataFrame:

    visualisation = TSNE(n_components=2).fit_transform(z_pca)
    df = pd.DataFrame(visualisation, columns = ["dimension 1", "dimension 2"])  #, "dimension 3"
    df["labels"] = labels
    return df

def plot_standard_devs(vae, data, labels, pca=None):
    # display a 2D plot showing the areas of impact of different points
    z_mean, z_log_var, z = vae.encoder.predict(data)
    if pca is None:
        pca = PCA()
        z_pca = pca.fit_transform(z_mean)
    else:
        z_pca = pca.transform(z_mean)

    z_log_var = pca.transform(z_log_var)

    plt.scatter(z_pca[:,0], latent_space_means[:,1], c = range(len(data)), label="means")
    plt.scatter(latent_space[:,0], latent_space[:,1], label="sampled", c = range(10), s=10)

    plt.legend()
    
    plt.show()
    
    # create 2 kernels
    m1 = (-1,-1)
    s1 = np.eye(2)
    k1 = multivariate_normal(mean=m1, cov=s1)

    m2 = (1,1)
    s2 = np.eye(2)
    k2 = multivariate_normal(mean=m2, cov=s2)

    # create a grid of (x,y) coordinates at which to evaluate the kernels
    xlim = (-3, 3)
    ylim = (-3, 3)
    xres = 100
    yres = 100

    x = np.linspace(xlim[0], xlim[1], xres)
    y = np.linspace(ylim[0], ylim[1], yres)
    xx, yy = np.meshgrid(x,y)

    # evaluate kernels at grid points
    xxyy = np.c_[xx.ravel(), yy.ravel()]
    zz = k1.pdf(xxyy) + k2.pdf(xxyy)

    # reshape and plot image
    img = zz.reshape((xres,yres))
    plt.imshow(img); plt.show()



def plot_label_clusters(vae, data, labels, pca=None, is_wae=False, top_5 = False):
    # display a 2D plot of the user classes in the latent space
    if is_wae:
        z_mean = vae.encoder.predict(data)
    else:
        z_mean, z_log_var, z = vae.encoder.predict(data)
    if top_5:
        z_mean = z_mean[:, :5]
    if pca is None:
        pca = PCA()
        z_pca = pca.fit_transform(z_mean)
    else:
        z_pca = pca.transform(z_mean)
    
    comps = 5

    pair_set = [(i,j) for i in range(comps) for j in range(i+1,comps)]
    
    fig, ax = plt.subplots(1 + len(pair_set)//3,3, figsize=(24,48))
    
    for ix, (i,j) in enumerate(pair_set):
        scatter = ax[divmod(ix, 3)].scatter(z_pca[:, i], z_pca[:, j], c=labels, alpha=0.7, cmap = "rainbow")
        ax[divmod(ix, 3)].set_xlabel(f"z[{i}]")
        ax[divmod(ix, 3)].set_ylabel(f"z[{j}]")
        legend = ax[divmod(ix, 3)].legend(*scatter.legend_elements(),
                    loc="upper right", title="Users")
        ax[divmod(ix, 3)].add_artist(legend)
    #plt.colorbar()
    
    plt.show()

def visualise(vae, data, labels, latent_dim=20, pca=None,
              is_wae=False, top_5=False):
    # display a 2D plot of the user classes in the latent space
    if is_wae:
        z_mean = vae.encoder.predict(data)
    else:
        z_mean, z_log_var, z = vae.encoder.predict(data)
    if top_5:
        z_mean = z_mean[:, :5]
    if pca is None:
        pca = PCA(min(z_mean.shape[1], 20))
        z_pca = pca.fit_transform(z_mean)
        #z_pca = z_mean
    else:
        z_pca = pca.transform(z_mean)
    
    df = dimension_reduction(z_pca, labels)

    fig, ax = plt.subplots(1,1, figsize=(24,12) ) # , subplot_kw={"projection": "3d"})# , 
                           
    ax.scatter(df["dimension 1"], df["dimension 2"],  c = df["labels"], cmap = "rainbow") #  df["dimension 3"],
    ax.set_xlabel("z[{i}]")
    ax.set_ylabel("z[{j}]")
    #plt.colorbar()
    
    plt.show()



"""    
intent_x = np.array(list(x_data[:1000])+list(x_data[-1000:]))
intent_y = np.array(list(y_intent[:1000]) + list(y_intent[-1000:]))

gesture_x = x_data[:600]
gesture_y = y_gesture.argmax(axis=1)[:600]

_map = (y_intent == 1) & ((y_user.argmax(axis=1) == 13) | (y_user.argmax(axis=1) == 14) | (y_user.argmax(axis=1) == 15))

user_x = x_data[_map]
user_y = y_user.argmax(axis=1)[_map]


#plot_label_clusters(vae, intent_x, intent_y, pca)
plot_label_clusters(vae, user_x, user_y, pca)
#plot_label_clusters(vae, gesture_x, gesture_y, pca)

visualise(vae, user_x, user_y, pca)
#visualise(vae, intent_x, intent_y)
#visualise(vae, gesture_x, gesture_y, pca)
"""


def plot_reconstructed_curves(vae, data, channel=0, is_wae=False):
    fig,ax = plt.subplots(2,5, figsize=(12,8))
    j = 0

    for i in random.sample(range(0, len(data)), 5):
        enc = vae.encoder.predict(data[i:i+1]) #vae.scaler.transform(
        if not is_wae:
            enc = enc[0] 
        dec = vae.scaler.inverse_transform(vae.decoder.predict(enc))
        ax[divmod(j,5)].plot(vae.scaler.inverse_transform(data)[i,:,channel])
        ax[divmod(j,5)].plot(dec[0,:,channel])

        j += 1

    plt.show()

    

"""
fig,ax = plt.subplots(2,8, figsize=(24,16))

filtered = x_data[y_intent==1]

for i in random.sample(range(0, len(filtered)), 1):
    for k in range(4):
        enc = vae.encoder.predict(filtered[i:i+1])
        dec = vae.decoder.predict(enc[2])
        for j in range(16):
            ax[divmod(j,8)].plot(dec[0,:,j])
            ax[divmod(j,8)].plot(dec[0,:,j])
    for j in range(16):
        ax[divmod(j,8)].plot(filtered[i,:,j], color="blue")
        

plt.show()


from random import randint

def average_dist(x, y):
    total = 0
    n = 1000
    for u in range(n):
        vx = np.array(x[randint(0, len(x)-1)])
        vy = np.array(y[randint(0, len(y)-1)])
        total += (sum((vx-vy)**2)**0.5)
    return total / n


a = np.zeros((16, 16))




for i in range(16):
    print(i)
    for j in range(i, 16):
        arr1, _, _ = vae.encoder(x_data[(y_intent == 1) & (y_user.argmax(axis=1) == i)])
        arr2, _, _ = vae.encoder(x_data[(y_intent == 1) & (y_user.argmax(axis=1) == j)])
        
        #print(f"User {i} compared to {j} : {average_dist(arr1, arr2)}")
        a[i,j] = average_dist(arr1, arr2)
        a[j,i] = average_dist(arr1, arr2)

plt.imshow(a, cmap='hot', interpolation='nearest')
plt.show()
"""
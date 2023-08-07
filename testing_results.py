import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
import tensorflow_ranking as tfr
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, LSTM, Permute, Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, Reshape, SpatialDropout1D, Dropout, TimeDistributed

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

import VAE
from VAE import Sampling, VAE, get_auth_model_from_latent_space, get_decoder, get_encoder, get_auth_model
from VAE_stats import VAE_stats
from scaler import CustomScaler


import datetime, csv, os, re, sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, f1_score, precision_score, precision_recall_curve, recall_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from scaler import CustomScaler

from featurize import filter, featurize
from sklearn.neural_network import MLPClassifier

#configs
maxprewindowsize = 4
classifier = 'rfc'

folds = 10


def get_average(l):
	return 0 if 0 == len(l) else sum(l) / len(l)

def get_eer(scores_legit, scores_adv):
	scores_legit = sorted(scores_legit)
	scores_adv = sorted(scores_adv)
	
	#treat each legitimate sample distance as a possible threshold, determine the point where FRR crosses FAR
	for c, threshold in enumerate(scores_legit):
		frr = c * 1.0 / len(scores_legit)
		adv_index = next((x[0] for x in enumerate(scores_adv) if x[1] > threshold), len(scores_adv))
		far = 1 - (adv_index * 1.0 / len(scores_adv))
		if frr >= far:
			return threshold, far
	print("Failure")

def get_eer_recogblind(scores_legit, scores_adv_typed, total_w, total_b, total_i):
	scores_legit = sorted(scores_legit)
	scores_adv_typed = sorted(scores_adv_typed, key = lambda x:x[0])
	
	#treat each legitimate sample distance as a possible threshold, determine the point where FRR crosses FAR
	for c, threshold in enumerate(scores_legit):
		frr = c * 1.0 / len(scores_legit)
		adv_index = next((x[0] for x in enumerate(scores_adv_typed) if x[1][0] > threshold), len(scores_adv_typed))
		far = 1 - (adv_index * 1.0 / len(scores_adv))
		if frr >= far:
			rejectrate_w = 0 if 0 == total_w else len([i for i in scores_adv_typed if 'W' == i[1] and i[0] >= threshold]) / total_w
			rejectrate_b = 0 if 0 == total_b else len([i for i in scores_adv_typed if 'B' == i[1] and i[0] >= threshold]) / total_b
			rejectrate_i = 0 if 0 == total_i else len([i for i in scores_adv_typed if 'I' == i[1] and i[0] >= threshold]) / total_i
			return threshold, far, rejectrate_w, rejectrate_b, rejectrate_i

def get_far_when_zero_frr(scores_legit, scores_adv):
	scores_legit = sorted(scores_legit)
	scores_adv = sorted(scores_adv)
	
	#treat each legitimate sample distance as a possible threshold, determine the point with the lowest FAR that satisfies the condition that FRR = 0
	for c, threshold in enumerate(scores_legit):
		frr = c * 1.0 / len(scores_legit)
		adv_index = next((x[0] for x in enumerate(scores_adv) if x[1] > threshold), len(scores_adv))
		far = 1 - (adv_index * 1.0 / len(scores_adv))
		if frr > 0.001:
			return threshold, far

def plot_threshold_by_far_frr(scores_legit, scores_adv, far_theta):
	scores_legit = sorted(scores_legit)
	scores_adv = sorted(scores_adv)
	
	frr = []
	far = []
	thresholds = []
	for c, threshold in enumerate(scores_legit):
		frr.append((c * 1.0 / len(scores_legit)) * 100)
		adv_index = next((x[0] for x in enumerate(scores_adv) if x[1] > threshold), len(scores_adv))
		far.append((1 - (adv_index * 1.0 / len(scores_adv))) * 100)
		thresholds.append(threshold)
	plt.figure(figsize = (6, 6))
	#plt.rcParams.update({'font.size': fontsize_legends})
	plt.plot(thresholds, far, 'tab:blue', label = 'FAR')
	plt.plot(thresholds, frr, 'tab:orange', label = 'FRR')
	plt.ylabel('error rate (%)')
	plt.xlabel(r'decision threshold, $\theta$')
	plt.axvline(x = far_theta, c = 'red')
	plt.legend(loc = 'best')
	plt.tight_layout(pad = 0.05)
	plt.show()

def plot_threshold_by_precision_recall(labels_test, labels_scores):
	p, r, thresholds = precision_recall_curve(labels_test, labels_scores)
	plt.figure(figsize = (6, 6))
	#plt.rcParams.update({'font.size': fontsize_legends})
	plt.title('Precision and Recall Scores as a Function of the Decision Threshold', fontsize = 12)
	plt.plot(thresholds, p[:-1], 'tab:blue', label = 'precision')
	plt.plot(thresholds, r[:-1], 'tab:orange', label = 'recall')
	plt.ylabel('score')
	plt.xlabel(r'decision threshold, $\theta$')
	plt.legend(loc = 'best')
	plt.tight_layout(pad = 0.05)
	plt.show()

def plot_roc_curve(labels_test, labels_scores):
	fpr, tpr, auc_thresholds = roc_curve(labels_test, labels_scores)
	print('AUC of ROC = ' + str(auc(fpr, tpr)))
	plt.figure(figsize = (6, 6))
	#plt.rcParams.update({'font.size': fontsize_legends})
	plt.title('ROC Curve', fontsize = 12)
	plt.plot(fpr, tpr, 'tab:orange', label = 'recall optimized')
	plt.plot([0, 1], [0, 1], 'k--')
	plt.axis([-0.005, 1, 0, 1.005])
	plt.xticks(np.arange(0, 1, 0.05), rotation = 90)
	plt.xlabel('false positive rate')
	plt.ylabel('true positive rate (recall)')
	plt.legend(loc = 'best')
	plt.tight_layout(pad = 0.05)
	plt.show()

def get_ascending_userID_list_string():
	for u in userIDs:
		if not 'user' in u and len(u) != 7:
			sys.exit('ERROR: userID not valid: ' + str(u))
	IDs = [int(u[4:]) for u in userIDs]
	IDs.sort(reverse = False)
	return ','.join([f'{i:03}' for i in IDs])

def get_descending_feature_list_string(weights, labels, truncate = 0):
	indicies = [i for i in range(len(weights))]
	for i in range(len(indicies)):
		for j in range(len(indicies)):
			if i != j and weights[indicies[i]] > weights[indicies[j]]:
				temp = indicies[i]
				indicies[i] = indicies[j]
				indicies[j] = temp
	if truncate != 0:
		del indicies[truncate:]
	return '\n'.join([str('%.6f' % weights[i]) + ' (' + labels[i] + ')' for i in indicies])

def write_verbose(f, s):
	outfilename = f + '-verbose.txt'
	outfile = open(outfilename, 'a')
	outfile.write(s + '\n')
	outfile.close()
    

class SplitLayer(tf.keras.layers.Layer):
    def __init__(self, layers):
        super(SplitLayer, self).__init__()
        self.layers = layers

    def call(self, inputs):
        return tf.gather(inputs,indices=self.layers,axis=-1)


def get_new_auth_model(input_dim=(200,16)):
    #input_dim = (200,16)
    
    inputs = keras.Input(shape=input_dim)
    x = inputs

    xs = []
    
    for i in range(16):
        
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

    x = Dense(25, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(10, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation="sigmoid")(x)
    #x = layers.Lambda(lambda x: x / 2)
    out = x

    model = keras.Model(inputs, out, name="LSTM_classifier_v2")
    
    return model


def run(models=["WatchAuth"], forbidden_stat = 0, users = range(16), limit_data_first_x = None, median_filtering= False, terminal_types = [3,4,5,6,7,8,9], score_mode= "macro", extra_data_handle=None, repetitions=1, save_name = "recent_results"):
    
    file_name = "raw_with_maps" # or offsets_2
    vae_stats = VAE_stats(10)
    
    if median_filtering:
        x_data = np.load(f"data/processed/x_{file_name}_median_filtered.npy")
        feature_x_data = np.load(f"data/processed/x_{file_name}_median_filtered_features.npy")
    else:
        x_data = np.load(f"data/processed/x_{file_name}_filtered.npy")  # _filtered
        feature_x_data = np.load(f"data/processed/x_{file_name}_features.npy")
    
    x_data = x_data[:, :, [0,1,2,3,4,5,6,7]]
    feature_x_data = vae_stats.vae_featurize(x_data)
    """
    print(feature_x_data.shape)
    allowed_channels = []
    for i in range(8*9):
        if i % 9 == forbidden_stat:
            allowed_channels.append(i)
    feature_x_data = feature_x_data.numpy()[:,allowed_channels]
    print(feature_x_data.shape)
    """
        
    y_user = np.load(f"data/processed/y_user_{file_name}.npy")
    y_intent = np.load(f"data/processed/y_intent_{file_name}.npy")
    y_gesture = np.load(f"data/processed/y_gesture_type_{file_name}.npy")

    train_gesture_map = np.load(f"data/processed/train_gesture_map_{file_name}.npy")
    test_gesture_map = np.load(f"data/processed/test_gesture_map_{file_name}.npy")
    
    feature_array = feature_x_data

    

    train_data_map = train_gesture_map.astype(bool)
    test_data_map = test_gesture_map.astype(bool)

    output = {}
    
    all_scores_legit = []
    all_scores_adv = []
    
    
    for auth_user in users:
        print(f"auth_user is {auth_user}")

        
        if not(limit_data_first_x is None):
            illegal_gestures = []
            
            for terminal_type in [3,4,5,6,7,8,9]:
                illegal_gesture_indices = (train_gesture_map==1) & (y_user.argmax(axis=1) == auth_user) & (y_gesture.argmax(axis=1) == terminal_type)

                illegal_gesture_indices = np.nonzero(illegal_gesture_indices)[0]
                illegal_gestures.append(illegal_gesture_indices[limit_data_first_x:])

            illegal_gestures = np.concatenate(illegal_gestures).astype(int)
            illegal_gesture_map = np.ones(len(train_gesture_map))
            
            illegal_gesture_map[illegal_gestures] = 0
            illegal_gesture_map = illegal_gesture_map.astype(bool)
            
            training_map = train_data_map & illegal_gesture_map
        
        else:
            training_map = train_data_map
            
        illegal_terminals = set([3,4,5,6,7,8,9]) - set(terminal_types)
        for terminal_type in illegal_terminals:
            training_map = training_map & ((y_gesture.argmax(axis=1) != terminal_type) | (y_user.argmax(axis=1) != auth_user ) )
            
        
        feature_data_train = feature_array[training_map]
        data_train = x_data[training_map]
        labels_train = (y_user.argmax(axis=1) == auth_user)[training_map].astype(int)
            
        
        if not (extra_data_handle is None):
            extra_x_data = np.load(extra_data_handle(auth_user))
            extra_x_data = extra_x_data # [:, :, [0,1,2,4,5,6]]
            
            labels_train = np.concatenate([labels_train, np.ones(len(extra_x_data))]) # change this
            
            if median_filtering:
                extra_x_feature_data = np.zeros((len(extra_x_data), feature_data_train.shape[1]))
                
                for i in range(0, len(extra_x_data), 128):
                    median_filtered_extra_data = tfa.image.median_filter2d(extra_x_data[i:i+128],filter_shape=(1,5), padding="constant")
                    extra_x_feature_data[i:i+128] = vae_stats.vae_featurize(median_filtered_extra_data)                
                
            else:
                extra_x_feature_data = np.zeros((len(extra_x_data), feature_data_train.shape[1]))
                extra_x_data = filter(extra_x_data, [0,1,2,3,4,5], 6, 3.667)
                for i in range(0, len(extra_x_data), 128):
                    extra_x_feature_data[i:i+128] = vae_stats.vae_featurize(extra_x_data[i:i+128])
                #extra_x_feature_data = featurize(extra_x_data)
            
            feature_data_train = np.concatenate([feature_data_train, extra_x_feature_data])
            data_train = np.concatenate([data_train, extra_x_data])
        
        print(np.unique(labels_train, return_counts=True))

        feature_data_test = feature_array[test_data_map]
        data_test = x_data[test_data_map]
        labels_test = (y_user.argmax(axis=1) == auth_user)[test_data_map].astype(int)
        
        #print(np.unique(labels_train, return_counts=True))
        
        for model_name in models:  # "DL", "MLP", "SVM", 

            key = str(auth_user) + "_" + model_name
            output[key] = {"fm":[], "prec":[], "rec":[], "eer":[], "far":[]}


            a_precisions = []
            a_recalls = []
            a_fmeasures = []
            a_pr_stdev = []
            a_re_stdev = []
            a_fm_stdev = []
            a_eers = []
            a_eer_thetas = []
            a_fars = []
            a_far_thetas = []
            a_ee_stdev = []
            a_ee_th_stdev = []
            a_fa_stdev = []
            a_fa_th_stdev = []
            a_aurocs = []
            a_losses = []


            
            if model_name in ["DL", "SVM"]:
                reps = 1
            else:
                reps = repetitions


            for repetition in range(reps):
                if model_name == "DL":

                    shuffled_data_train, shuffled_labels_train = shuffle(data_train, labels_train, random_state=0)

                    #ORDER = 6
                    #CUTOFF = 3.667
                    #shuffled_data_train = filter(shuffled_data_train, list(range(16)), ORDER, CUTOFF)

                    scaler = CustomScaler()
                    shuffled_data_train = scaler.fit_and_transform(shuffled_data_train)


                    kFold = sklearn.model_selection.StratifiedKFold(n_splits=5)
                    val_map = next(kFold.split(shuffled_data_train, shuffled_labels_train))[1]


                    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', start_from_epoch=50,
                                                                               patience=50, restore_best_weights=True)


                    model = None
                    model = get_new_auth_model((200,16))
                    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss="binary_crossentropy")
                    history = model.fit(shuffled_data_train[~val_map], shuffled_labels_train[~val_map], epochs=1000,
                                        batch_size=128, verbose=1,
                                        validation_data=(shuffled_data_train[val_map], shuffled_labels_train[val_map]),
                                       class_weight={0:0.5, 1:30},
                                       callbacks=[early_stopping_callback])

                    plt.plot(history.history['loss'])
                    plt.plot(history.history['val_loss'])
                    plt.title('model loss')
                    plt.ylabel('loss')
                    plt.xlabel('epoch')
                    plt.legend(['train', 'val'], loc='upper left')
                    plt.show()

                    plt.clf()

                    labels_scores = model.predict(scaler.transform(data_test)).squeeze()
                    labels_pred = (labels_scores >= 0.5).astype(int)



                elif model_name == "WatchAuth":
                    model = RandomForestClassifier(n_estimators = 100, random_state = repetition, class_weight="balanced").fit(feature_data_train, labels_train)


                    labels_pred = model.predict(feature_data_test)
                    labels_scores = model.predict_proba(feature_data_test)[:, 1]

                elif model_name == "SVM":

                    clf = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True, class_weight="balanced"))
                    clf.fit(feature_data_train, labels_train)

                    labels_pred = clf.predict(feature_data_test)
                    labels_scores = clf.predict_proba(feature_data_test)[:, 1]

                elif model_name == "MLP":

                    clf = make_pipeline(StandardScaler(), MLPClassifier(solver='lbfgs', alpha=1e-3,
                             hidden_layer_sizes=(50,20, 2), random_state=repetition)) # ,class_weight="balanced")) 
                    clf.fit(feature_data_train, labels_train)

                    labels_pred = clf.predict(feature_data_test)
                    labels_scores = clf.predict_proba(feature_data_test)[:, 1]

                #get precision, recall, and F-measure scores
                precision = precision_score(labels_test, labels_pred, average = score_mode, labels = np.unique(labels_pred))
                recall = recall_score(labels_test, labels_pred, average = score_mode, labels = np.unique(labels_pred))
                fmeasure = f1_score(labels_test, labels_pred, average = score_mode, labels = np.unique(labels_pred)) # macro vs binary
                auroc = sklearn.metrics.roc_auc_score(labels_test, labels_scores, average = "macro", labels = np.unique(labels_pred))
                a_precisions.append(precision)
                a_recalls.append(recall)
                a_fmeasures.append(fmeasure)
                a_aurocs.append(auroc)
                a_losses.append(sklearn.metrics.log_loss(labels_test, labels_scores))
                #print(fmeasure)

                #print(f"Loss is {sklearn.metrics.log_loss(labels_test, labels_scores)}")
                #print(precision)
                #print(recall)


                scores_legit = [labels_scores[i] for i in range(len(labels_test)) if 1 == labels_test[i]]
                scores_adv = [labels_scores[i] for i in range(len(labels_test)) if 0 == labels_test[i]]
                eer_theta, eer = get_eer(scores_legit, scores_adv)
                far_theta, far = get_far_when_zero_frr(scores_legit, scores_adv)

                #plot_roc_curve(labels_test, labels_scores)
                #plot_threshold_by_precision_recall(labels_test, labels_scores)
                #plot_threshold_by_far_frr(scores_legit, scores_adv, far_theta)

                #print(f"eer is {eer}")
                a_eers.append(eer)
                a_eer_thetas.append(eer_theta)
                a_fars.append(far)
                a_far_thetas.append(far_theta)

                all_scores_legit += scores_legit
                all_scores_adv += scores_adv


            a_pr_stdev = np.std(a_precisions, ddof = 1)
            a_re_stdev = np.std(a_recalls, ddof = 1)
            a_fm_stdev = np.std(a_fmeasures, ddof = 1)
            a_ee_stdev = np.std(a_eers, ddof = 1)
            a_ee_th_stdev = np.std(a_eer_thetas, ddof = 1)
            a_fa_stdev = np.std(a_fars, ddof = 1)
            a_fa_th_stdev = np.std(a_far_thetas, ddof = 1)
            
            a_auroc_stdev = np.std(a_aurocs, ddof = 1)
            a_losses_stdev = np.std(a_losses, ddof = 1)

            output[key]["fm"] = (get_average(a_fmeasures), a_fm_stdev)
            output[key]["prec"] = (get_average(a_precisions), a_pr_stdev)
            output[key]["rec"] = (get_average(a_recalls), a_re_stdev)
            output[key]["eer"] = (get_average(a_eers), a_ee_stdev)
            output[key]["far"] = (get_average(a_fars), a_fa_stdev)
            output[key]["auroc"] = (get_average(a_aurocs), a_auroc_stdev)
            output[key]["loss"] = (get_average(a_losses), a_losses_stdev)


            with open(f'data/stats/{save_name}.pickle', 'wb') as handle:
                pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)


    # multiple_models_scaled_v2.pickle was a good summary of different types of model

    return output, (all_scores_legit, all_scores_adv)

def summarise_by_model(output, models=["WatchAuth"]):
    
    for stat in ["fm", "prec", "rec", "eer", "far", "auroc", "loss"]:
        for model_name in models:
            l = []
            for auth_user in range(16):
                l.append(output[f"{auth_user}_{model_name}"][stat][0])
            print(model_name, stat, sum(l) /16)
        print("")
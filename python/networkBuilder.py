# TensorFlow and tf.keras
import tensorflow as tf

import keras
import keras.backend
from keras import layers
from keras import models
from keras import optimizers
import keras.utils

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import sklearn
from sklearn.utils import shuffle

def build_DNN(X_train):
    input2 = [layers.Input(shape = (len(X_train[i][0]),)) for i in range(0,len(X_train))]
    if len(input2) == 1:
        x = layers.Dense(64,activation='relu')(input2[0])
    else:
        x = layers.concatenate(inputs = input2, axis = -1)
        x = layers.Dense(64, activation = 'relu')(x)
    x = layers.Dense(128, activation = 'relu')(x)
    output = layers.Dense(2, activation = 'softmax')(x)
    model = models.Model(inputs = input2, 
                         outputs = output)
    opt=keras.optimizers.Adam(lr = 0.0005,
                              beta_1 = 0.9,
                              beta_2 = 0.9,
                              amsgrad = False)
    model.compile(loss = 'categorical_crossentropy',
                optimizer = opt,
                metrics = ['categorical_crossentropy', 'accuracy'])
    return model

def build_CNN_2D(X_train):
    input2 = [layers.Input(shape = (len(X_train[i][0]),)) for i in range(1,len(X_train))]
    input1 = layers.Input(shape = (grid, grid,1))
    x = layers.Conv2D(32, (5, 5), activation = 'relu', padding = 'same')(input1)
    x = layers.Conv2D(32, (3, 3), activation = 'relu', padding = 'same')(x)
    x = layers.Conv2D(32, (2, 2), activation = 'relu', padding = 'same')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.MaxPool2D((2, 2))(x)
    x1 = layers.Flatten()(x)
    if input2 == []:
        x = layers.Dense(64, activation='relu', name = 'dense1')(x1)
    else:
        x = layers.concatenate(inputs = [x1] + input2, axis = -1)
        x = layers.Dense(64, activation='relu', name = 'dense2')(x)
    x = layers.Dense(128, activation='relu', name = 'dense3')(x)
    output = layers.Dense(2, activation='softmax')(x)
    model = models.Model(inputs= [input1] + input2,
                         outputs = output)
    opt = keras.optimizers.Adam(lr = 0.001,
                                beta_1 = 0.9,
                                beta_2 = 0.999,
                                amsgrad = False)
    model.compile(loss = 'categorical_crossentropy',
                optimizer = opt,
                metrics = ['categorical_crossentropy', 'accuracy'])
    return model

def build_XY(features, label, dic1, dic2, signal="Zbb"):
    X = [np.concatenate((dic1[key], dic2[key])) for key in features]
    Y = [np.concatenate((dic1[key], dic2[key])) for key in label]
    dim = [ele.shape+(1,) for ele in X]
    for i in range(0,len(features)):
        X[i] = X[i].reshape(dim[i])

    #choose the correct column for signal labels                               
    if "Zbb" in signal:
        Y = [Y[0][:,::2]]
    else:
        Y = [Y[0][:,:2]]
        
    #Randomize!                                                                
    np.random.seed(1)
    # split signal and background indices                                      
    ind = np.argwhere(Y[0])[:,0]
    # shuffle indices randomly                                                 
    np.random.shuffle(ind)
    for j in range(0,len(features)):
        X[j] = X[j][ind]
    Y = [Y[0][ind]]

    return X,Y


def get_feat(flav=True,  images=False, particleInfo=False):
    feat = ['jetPt',
            'jetEta',
            'jetPhi',
            'jetMass',
            'jetMassSD',
            'deltaR_subjets',
            'z',
            'tau1_b05',
            'tau2_b05',
            'tau3_b05',
            'tau1_sd_b05',
            'tau2_sd_b05',
            'tau3_sd_b05',
            'tau1_b10',
            'tau2_b10',
            'tau3_b10',
            'tau1_sd_b10',
            'tau2_sd_b10',
            'tau3_sd_b10',
            'tau1_b20',
            'tau2_b20',
            'tau3_b20',
            'tau1_sd_b20',
            'tau2_sd_b20',
            'tau3_sd_b20',
            'beta3',
            'beta3_sd',
            'tau21',
            'jetpull_abs']

    if flav:
        feat.extend(['chMult',
                     'neutMult',
                     'phoMult',
                     'eleMult',
                     'muMult',
                     'dxy_max',
                     'dz_max'])

    if images:
        feat.insert(0,'jetImages')

    if particleInfo:
        feat.extend(['jetconstPt_log',
                     'jetconstEta_abs',
                     'jetconstE_log',
                     'jetconstPt_Jetlog',
                     'charge',
                     'isEle',
                     'isPho',
                     'isMuon',
                     'isCh',
                     'isNh',
                     'delta_eta',
                     'delta_phi',
                     'deltaR_jet',
                     'deltaR_subjet0',
                     'deltaR_subjet1',
                     'dxy',
                     'dz'])
    return feat


# TensorFlow and tf.keras
import tensorflow as tf
#tf.enable_eager_execution()

from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd

import keras
import keras.backend
from keras import layers
from keras import models
import keras.utils

# Helper libraries
import numpy as np
from scipy import stats,special

from matplotlib.colors import LogNorm
from matplotlib import gridspec
import math
import time
import h5py
import sklearn
from sklearn.utils import shuffle

# innvestigate for lrp
import innvestigate
import innvestigate.utils as iutils
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
import matplotlib.cm

import os
import datetime


def printTime(delta, definition='Elapsed Time'):
    
    
    time_string = '{0:0.1f} h {1:0.1f} min {2:0.2f} sec'.format(delta // 3600, delta % 3600 // 60, delta % 3600 % 60)
    full_string = definition + ': ' + time_string
    print(full_string)
    


def get_lrp_score(test_inputs, model, nXvar, totalVar, batchShape=[20]):
    
    
    # test_inputs is X_test output from build_XY function in networkBuilder.py 
    # model loaded from model = keras.models.load_model(model_name)
    # nXvar = number of expert features (xaugs)
    # totalVar = total number of features
    # batchShape =  [20] for particle list: number of constituent particles per event
    #               [16, 16] for jet images: grid shape
    
    lrp_toc = time.time()
    
    # strip model of softmax layer
    best_model_wo_softmax = innvestigate.utils.keras.graph.model_wo_softmax(model)
     
    # build lrp analyzer using Preset A: LRP_alpha1beta0 for conv layers and LRP_epsilon for dense layers
    lrp_analyzer = innvestigate.analyzer.relevance_based.relevance_analyzer.LRPSequentialPresetA(best_model_wo_softmax)
   
    # run lrp analysis in chunks so that kernel does not die
    nElem = 10000
    batchsize = test_inputs[0].shape[0]
    slices = batchsize // nElem
    size_from_loop = batchsize - (batchsize % nElem)
    
    print()
    print('Running LRP Analysis')
    print('Total Events: {0:0.0f}'.format(batchsize))
    print('Split into chunk size: {0:0.0f}'.format(nElem))
    print()
    
    lrp_list = []
    

    # get lrp scores from all chunks except last chunk
    for i in range(slices):

        lrp_i = lrp_analyzer.analyze([element[nElem*i:nElem*(i+1)] for element in test_inputs])
        lrp_list.append(lrp_i)
        del lrp_i

    
    # get lrp scores from last chunk
    lrp_list_last = lrp_analyzer.analyze([element[size_from_loop:] for element in test_inputs])
    
    # combine chunks into separate arrays for particle list features (lrp_plist) and xaug features (lrp_xaugs)
    lrp_plist_list = []
    lrp_xaugs_list = []
    
    lrp_plist_list_last = []
    lrp_xaugs_list_last = []


    # append lrp scores to lrp_plist or lrp_xaugs
    for i in range(totalVar):
        for s in range(slices):
            if(i < (totalVar-nXvar)):
                lrp_plist_list.append(lrp_list[s][i])
                    
            else:
                lrp_xaugs_list.append(lrp_list[s][i])
                
    # append lrp scores from last chunk         
    for i in range(totalVar):
        if(i < (totalVar-nXvar)):
            lrp_plist_list_last.append(lrp_list_last[i])

        else:
            lrp_xaugs_list_last.append(lrp_list_last[i])
                
                
    # copy to arrays           
    lrp_plist_first = np.array(lrp_plist_list).reshape(totalVar-nXvar, size_from_loop, *batchShape, 1)
    lrp_xaugs_first = np.array(lrp_xaugs_list).reshape(nXvar, size_from_loop, 1)
    
    lrp_plist_last = np.array(lrp_plist_list_last).reshape(totalVar-nXvar,(batchsize % nElem),*batchShape,1)
    lrp_xaugs_last = np.array(lrp_xaugs_list_last).reshape(nXvar,(batchsize % nElem),1)
                
    
    # append last chunk to lrp scores arrays
    lrp_plist = np.append(lrp_plist_first, lrp_plist_last, axis=1)
    lrp_xaugs = np.append(lrp_xaugs_first, lrp_xaugs_last, axis=1)

    del lrp_plist_list
    del lrp_xaugs_list
    del lrp_plist_list_last
    del lrp_xaugs_list_last
    del lrp_list
    del lrp_plist_first
    del lrp_xaugs_first
    del lrp_plist_last
    del lrp_xaugs_last
    
    lrp_tic = time.time()
    
    print()
    printTime(lrp_tic - lrp_toc, 'LRP Analysis Time')
    
    return lrp_plist, lrp_xaugs
    

def get_normalized_lrp_score(lrp_plist, lrp_xaugs):

    lrp_plist_norm = np.zeros_like(lrp_plist)
    lrp_xaugs_norm = np.zeros_like(lrp_xaugs)
    
    if(len(lrp_plist > 0)):
        for i in range(lrp_plist.shape[1]):

            maxval = np.max(abs(lrp_plist[:,i].flatten()))
            if(maxval < 1e-6): maxval = 1.
            lrp_plist_norm[:,i] = lrp_plist[:,i] / maxval

    if(len(lrp_xaugs > 0)):   
        for i in range(lrp_xaugs.shape[1]):

            maxval = np.max(abs(lrp_xaugs[:,i].flatten()))
            lrp_xaugs_norm[:,i] = lrp_xaugs[:,i] / maxval
    
    
    return lrp_plist_norm, lrp_xaugs_norm

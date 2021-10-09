# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:41:27 2021

@author: haako
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.utils import resample
from sklearn.model_selection import KFold
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot
from sklearn import linear_model
from imageio import imread
from time import time

#Standardize features by removing the mean and scaling to unit variance.
def scalerStandard(X_train, X_test, z_train, z_test):
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #scale the response variable
    z_train_scaled = (z_train - np.mean(z_train))/np.std(z_train)
    z_test_scaled = (z_test - np.mean(z_train))/np.std(z_train)
    '''
    X_train_scaled = scaler.fit_transform(X_train, with_std = False) #æææ
    X_test_scaled = scaler.fit_transform(X_test)
    z_train_scaled = (z_train - np.mean(z_train))/np.std(z_train)
    z_test_scaled = (z_test - np.mean(z_test))/np.std(z_test)
    '''

    return X_train_scaled, X_test_scaled, z_train_scaled, z_test_scaled

def scalerMean(X_train, X_test, z_train, z_test):
    mean_X = np.mean(X_train, axis=0)
    X_train_scaled = X_train - mean_X
    X_test_scaled = X_test - mean_X
    mean_z = np.mean(z_train)
    z_train_scaled = z_train - mean_z
    z_test_scaled = z_test - mean_z
    
    
    return X_train_scaled, X_test_scaled, z_train_scaled, z_test_scaled

def scalerNone(X_train, X_test, z_train, z_test):
    return X_train, X_test, z_train, z_test

#Scales the features to a given range, here [0,1]
def scalerMinMax(X_train, X_test, z_train, z_test):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    #Scale the response variable
    z_train = z_train.reshape((-1,1))
    z_test = z_test.reshape((-1,1))
    scaler = MinMaxScaler(feature_range=(0,1)).fit(z_train)
    z_train_scaled = scaler.transform(z_train)
    scaler = MinMaxScaler(feature_range=(0,1)).fit(z_train)
    z_test_scaled = scaler.transform(z_test)

    return X_train_scaled, X_test_scaled, z_train_scaled.flatten(), z_test_scaled.flatten()

# #Scales samples so that L2 = 1
# def scalerNormalizer(X_train, X_test, z_train, z_test):
#     scaler = Normalizer().fit(X_train)
#     X_train_scaled = scaler.transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     #Scale the response variable
#     z_train = z_train.reshape((1,-1))
#     z_test = z_test.reshape((1,-1))
#     scaler = Normalizer().fit(z_train)
#     z_train_scaled = scaler.transform(z_train)
#     scaler = Normalizer().fit(z_train)
#     z_test_scaled = scaler.transform(z_test)

#     return X_train_scaled, X_test_scaled, z_train_scaled.flatten(), z_test_scaled.flatten()

#Scale features using statistics that are robust to outliers
def scalerRobust(X_train, X_test, z_train, z_test):
    scaler = RobustScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #Scale the response variable
    z_train = z_train.reshape((-1,1))
    z_test = z_test.reshape((-1,1))
    scaler = RobustScaler().fit(z_train)
    z_train_scaled = scaler.transform(z_train)
    scaler = RobustScaler().fit(z_train)
    z_test_scaled = scaler.transform(z_test)

    return X_train_scaled, X_test_scaled, z_train_scaled.flatten(), z_test_scaled.flatten()

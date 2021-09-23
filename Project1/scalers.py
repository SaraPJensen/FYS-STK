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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.utils import resample


def scalerStandard(X_train, X_test, z_train, z_test):
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #scale the response variable
    z_train_scaled = (z_train - np.mean(z_train))/np.std(z_train)
    z_test_scaled = (z_test - np.mean(z_train))/np.std(z_train)

    return X_train_scaled, X_test_scaled, z_train_scaled, z_test_scaled

def scalerMinMax(X_train, X_test, z_train, z_test):
    scaler = MinMaxScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #Scale the response variable
    z_train = z_train.reshape((-1,1))
    z_test = z_train.reshape((-1,1))
    scaler2 = MinMaxScaler()
    scaler3 = MinMaxScaler()
    scaler2.fit(z_train)
    scaler3.fit(z_test)
    z_train_scaled = scaler2.transform(z_train)
    z_test_scaled = scaler3.transform(z_test)

    return X_train_scaled, X_test_scaled, z_train_scaled.flatten(), z_test_scaled.flatten()

def scalerNormalizer(X_train, X_test, z_train, z_test):
    scaler = Normalizer().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #Scale the response variable
    z_train = z_train.reshape((1,-1))
    z_test = z_test.reshape((1,-1))
    scaler = Normalizer().fit(z_train)
    z_train_scaled = scaler.transform(z_train)
    scaler = Normalizer().fit(z_test)
    z_test_scaled = scaler.transform(z_test)

    return X_train_scaled, X_test_scaled, z_train_scaled.flatten(), z_test_scaled.flatten()

def scalerRobust(X_train, X_test, z_train, z_test):
    scaler = RobustScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #Scale the response variable
    z_train = z_train.reshape((-1,1))
    z_test = z_test.reshape((-1,1))
    scaler = RobustScaler().fit(z_train)
    z_train_scaled = scaler.transform(z_train)
    scaler = RobustScaler().fit(z_test)
    z_test_scaled = scaler.transform(z_test)

    return X_train_scaled, X_test_scaled, z_train_scaled.flatten(), z_test_scaled.flatten()

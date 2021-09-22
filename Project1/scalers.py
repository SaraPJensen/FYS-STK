# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:41:27 2021

@author: haako
"""

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

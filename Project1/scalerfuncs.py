# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 17:39:59 2021

@author: haako
"""

import numpy as np
import matplotlib.pyplot as plt



from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.model_selection import train_test_split



def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def design_matrix(x, y, poly):
    l = int((poly+1)*(poly+2)/2)		# Number of elements in beta
    X = np.ones((len(x),l))

    for i in range(1,poly+1):
        q = int((i)*(i+1)/2)

        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k)
    return X

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
    scaler = Normalizer(norm='max')
    


n = 2
poly = 5

x = np.sort(np.random.uniform(0, 1, n))
y = np.sort(np.random.uniform(0, 1, n))
x, y = np.meshgrid(x, y)
z = FrankeFunction(x, y) + 0.1*np.random.randn(n, n)
x = np.ravel(x)
y = np.ravel(y)
z = np.ravel(z)
X = design_matrix(x, y, poly)
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size = 0.2)

X_train_scaled, X_test_scaled, z_train_scaled, z_test_scaled = scalerMinMax(X_train, X_test, z_train, z_test)

print("x_train:\n", z_train)
print("X_train_scaled:\n", z_train_scaled)




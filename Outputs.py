# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 15:12:59 2020

@author: shush
"""

import numpy as np

ncut=20
test_size=80
nNodes=128
min_tol=0.01

output=np.load('HR_128_512/Results/OutProb-NMF-HR_128_512.npy',allow_pickle='TRUE').item()
NMF_prob=np.empty(shape=[test_size,ncut+1,nNodes], dtype='float32')
for i in range(ncut+1):
    NMF_prob[:,i,:]=output[i]
NMF_inf=np.sum(NMF_prob,2)

WeightBias=np.load('HR_128_512/Results/WeightBias-NMF-HR_128_512.npy',allow_pickle='TRUE').item()#all weights and biases
predicted_A=WeightBias['layer/weightA']
predicted_A=predicted_A*(predicted_A>min_tol)

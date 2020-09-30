# -*- coding: utf-8 -*-
"""
To get the train data.
Created on Mon Feb 24 03:40:34 2020
@author: shushan
"""
#%%
import numpy as np


nNodes=128
test_size=100

def to_nparray(dataset,nNodes):
    row_size=len(dataset)
    cascades_time=np.full((row_size,nNodes),np.inf)
    for i in range(row_size):
        cascade=np.fromstring(dataset[i],dtype=float,sep=';')
        for j in range(int(len(cascade)/2)):
            cascades_time[i,int(cascade[2*j])]=cascade[2*j+1]
    return cascades_time  

def to_npy(dataset,tcut,ncut,timepoints):
    nsamples=dataset.shape[0]
    nnodes=dataset.shape[1]
    cascades=np.empty([nsamples,ncut+1,nnodes])
    if timepoints==None:
        dt=tcut/ncut
        t=np.arange(0,tcut+dt,dt)
    else:
        t=timepoints
    for i in range(ncut+1):
        cascades[:,i,:]=np.multiply((dataset<=t[i]), 1)
    return cascades


class clean_data_npy:
    def __init__(self,filenumber,Keywords,tcut,ncut,timepoints):
        self.filenumber=filenumber
        self.Keywords=Keywords       
        self.Datadir=Keywords+'/Datasets'

        self.nNodes=128
        self.tcut=tcut
        self.ncut=ncut
        self.timepoints=timepoints
        
        self.traintxt_file=self.Datadir+'/cascades-train-'+Keywords+'-'+str(filenumber)  
        self.trainnpy_file=self.Datadir+'/cascades-train-'+str(filenumber)+'.npy'

    def train_data(self):
        train_cascades=[]
        f=open(self.traintxt_file,"r+")
        for line in f:
            parts=line.replace("\n","")
            train_cascades.append(parts)
        f.close()
        train_cascades_time=to_nparray(train_cascades,self.nNodes)
        train_data=to_npy(train_cascades_time,self.tcut,self.ncut,self.timepoints)
        np.save(self.trainnpy_file,train_data)           
        return 
    
 
######################### main #####################################################
def main(): 
    tcut=20
    ncut=20
    Keywords='HR_128_512'
    for filenumber in range(1,2):
        data=clean_data_npy(filenumber,Keywords,tcut,ncut,None)
        data.train_data()
        
main()
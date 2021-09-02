#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 15:19:02 2021

@author: randallclark
"""


import numpy as np
from numba import njit
from sklearn.neighbors import KDTree


class RBF_NN:
    '''
    This version of DDF_RBF includes a slight adaption that doesn't choose a set R for all gaussians, but rather, it tries to find a
    choice of R that is relative to its distance to its neighbors. The goal of this is to reduce overlap between centers while not creating
    vast amounts of empty space in the nighborhood of less dense areas.
    
    
    Build a function that approximates the behavior of F(t)=dx/dt for all dimensions D
    inputs:
        Xdata - This is the training data, must be D by T format (where D is dimensions and T is the total training length)
        length - The Training length
        p - the number of steps to take before adding a training data point to the list of centers (i.e. if p = 5, every 5 data points will be a center)
        beta - Regularization paramemter in Ridge Regression
        D - Number of Dimensions in the model being trained on
        b - this is the number of nearest neighbors you wish to average over to create the relative R value
    '''    
    def FuncApproxF(self,Xdata,length,p,beta,R,D,b):
        #To Create the F(x) we will need only X, generate Y, and perform Ridge Regression to build the Function
        #Xdata will need to be 1 point longer than length to make Y
        #Make sure both X and Y are in D by T format
        Ydata = self.GenY(Xdata,length,D)
    
        #Here you choose how you will define your ceneters:
        #C = X  #Basic Centers
        XdataT = Xdata.T
        NcLength = int(length/p)
        C = np.zeros((NcLength,D))
        for i in range(NcLength):
            C[i] = XdataT[i*p] #Here we choose every p points from Xdata to make a center
        
        #Here We will construct our R matrix from. 
        RR = self.CreateRMat(C,b)
        
        #Creat the Phi matrix with those centers
        PhiMat = self.CreatePhiMat(XdataT,C,length,NcLength,RR,R)
        
        
        #Perform RidgeRegression
        YPhi = np.zeros((D,NcLength))
        for d in range(D):
            YPhi[d] = np.matmul(Ydata[d],PhiMat.T)
        PhiPhi = np.linalg.inv(np.matmul(PhiMat,PhiMat.T)+beta*np.identity(NcLength))
        W = np.zeros((D,NcLength)) 
        for d in range(D):
            W[d] = np.matmul(YPhi[d],PhiPhi)
        
        #Now we want to put together a function to return
        @njit
        def F(x):
            f = np.zeros(D)
            for d in range(D):
                for c in range(NcLength):
                    f[d] = f[d] + W[d][c]*np.exp(-(np.linalg.norm(x-C[c])**2)*R*RR[c])
            return f
        
        
        self.FinalX = Xdata.T[length-1]
        self.D = D
        return F
        
    
    '''
    Predict Forward in Time
    inputs:
        F - This is the Function created above, simply take the output from the above section and input it into this one
        PreLength - Choose how long you want to predict for
        StartPoint - We can choose to predict from anywhere, although usually I choose to start predicting where the training window ends
    '''  
    def PredictIntoTheFuture(self,F,PreLength,StartPoint):
        @njit
        def makePre(D,FinalX):
            Prediction = np.zeros((PreLength,D))
            #Start by Forming the Bases
            Prediction[0] = FinalX+F(FinalX)
            
            #Let it run forever now
            for t in range(1,PreLength):
                Prediction[t] = Prediction[t-1]+F(Prediction[t-1])
            return Prediction
        
        Prediction = makePre(self.D,StartPoint)
        return Prediction.T


        
    """
    These three functions are Secondary functions used in the F function creation process, feel free to ignore them unless you which to
    change the code. I seperate them down here to make the above functions cleaner.
    """
    def CreatePhiMat(self,X,C,Nlength,NcLength,RR,R):
        @njit
        def getMat():
            Mat = np.zeros((NcLength,Nlength))
            for i in range(NcLength):
                for j in range(Nlength):
                    Mat[i][j] = np.exp(-(np.linalg.norm(X[j]-C[i])**2)*R*RR[i])
            return Mat
        Mat = getMat()
        return Mat


    def GenY(self,Xdata,length,D):
        @njit
        def makeY():
            Y = np.zeros((D,length))
            for d in range(D):
                for t in range(length):
                    Y[d][t] = Xdata[d][t+1]-Xdata[d][t]
            return Y
        Y = makeY()
        return Y
    
    def CreateRMat(self,centers,b):
        kdt = KDTree(centers, leaf_size=10, metric='euclidean')
        dist, ind = kdt.query(centers, k=b+1, return_distance=True)
        A = np.zeros(len(dist))
        for i in range(b):
            A = A + dist.T[i+1]
        A = A/b
        for i in range(len(A)):
            A[i] = 1/A[i]
        return A
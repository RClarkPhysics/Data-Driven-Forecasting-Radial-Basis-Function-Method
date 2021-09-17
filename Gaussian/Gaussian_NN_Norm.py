#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 15:29:27 2021

@author: randallclark
"""


import numpy as np
from numba import njit
from sklearn.neighbors import KDTree


class RadialBExp:
    '''
    Initialize the Class
    Parametrs:
        name - The name of the Data set being trained on
        D - The number of dimensions of the data set being trained on and predicted
        dt - the time step
    '''
    def __init__(self,D,dt):
        self.D = D
        self.dt = dt
    
    def FuncApproxF(self,Xdata,length,p,beta,alpha,b):
        self.DData = Xdata.copy()
        self.Scaler = np.zeros(self.D)
        for d in range(self.D):
            self.Scaler[d] = abs(max(Xdata[d])-min(Xdata[d]))
            self.DData[d] = self.DData[d]/self.Scaler[d]
        
        
        #To Create the F(x) we will need only X, generate Y, give that to the Func Approxer
        #Xdata will need to be 1 point longer than length to make Y
        #Make sure both X and Y are in D by T format
        Ydata = self.GenY(self.DData,self.dt,length,self.D)
        F = self.CreateFunction(Ydata,self.DData,length,p,beta,b)
        
        self.FinalX = np.zeros(self.D)
        for d in range(self.D):
            self.FinalX[d] = self.DData[d][length-1]
        self.Ydata = Ydata
        
        return F
        


    def CreateFunction(self,Ydata,Xdata,Nlength,p,beta,b):
        #The goal of this section is to take a set of data and a radial basis expansion
        #Then to train the Function to output the values of the unknown function that generated the data
        #Training will be performed using Ridge Regression
        NcLength = int(Nlength/p)
        
        Ytarget = np.zeros((self.Dy,Nlength))
        X = np.zeros((self.Dx,Nlength))
        for i in range(self.Dy):
            Ytarget[i] = Ydata[i][0:Nlength]
        for i in range(self.Dx):
            X[i] = Xdata[i][0:Nlength]
        X = X.T
        
        #Here you choose how you will define your ceneters:
        #C = X  #Basic Bitch Centers
        C = np.zeros((NcLength,self.Dx))
        for i in range(NcLength):
            C[i] = X[i*p]
            
        #Here We will construct our R matrix from    
        RR = self.CreateRMat(C,b)
        
        #Creat the Phi matrix with those centers
        PhiMat = self.CreatePhiMat(X,C,Nlength,NcLength,RR)
        
        #Perform RidgeRegression
        YPhi = np.zeros((self.Dy,NcLength))
        for d in range(self.Dy):
            YPhi[d] = np.matmul(Ytarget[d],PhiMat.T)
        PhiPhi = np.linalg.inv(np.matmul(PhiMat,PhiMat.T)+beta*np.identity(NcLength))
        W = np.zeros((self.Dy,NcLength)) 
        for d in range(self.Dy):
            W[d] = np.matmul(YPhi[d],PhiPhi)
            
        #Now we want to put together a function to return
        DD = self.Dy
        alpha = self.alpha
        @njit
        def newfunc(x):
            f = np.zeros(DD)
            for d in range(DD):
                for c in range(NcLength):
                    f[d] = f[d] + W[d][c]*np.exp(-(np.linalg.norm(x-C[c])**2)*RR[c]*alpha)
            return f
        return newfunc
        
    
    def CreatePhiMat(self,X,C,Nlength,NcLength,RR):
        @njit
        def getMat(alpha):
            Mat = np.zeros((NcLength,Nlength))
            for i in range(NcLength):
                for j in range(Nlength):
                    Mat[i][j] = np.exp(-(np.linalg.norm(X[j]-C[i])**2)*RR[i]*alpha)
            return Mat
        Mat = getMat(self.alpha)
        return Mat
        
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
    
    
    
    
    
    def PredictIntoTheFuture(self,F,XPredict,PreLength):
        @njit
        def makePre(D,FinalX,dt):
            Prediction = np.zeros((PreLength,D))
            #Start by Forming the Bases
            Prediction[0] = FinalX+dt*F(FinalX)
            
            #Let it run forever now
            for t in range(1,PreLength):
                Prediction[t] = Prediction[t-1]+dt*F(Prediction[t-1])
            return Prediction
        
        Prediction = makePre(self.D,self.FinalX,self.dt)
        Prediction = Prediction.T
        
        for d in range(self.D):
            Prediction[d] = Prediction[d]*self.Scaler[d]
        return Prediction





    def GenY(self,Xdata,dt,length,D):
        @njit
        def makeY():
            Y = np.zeros((D,length))
            for d in range(D):
                for t in range(length):
                    Y[d][t] = (Xdata[d][t+1]-Xdata[d][t])/dt
            return Y
        Y = makeY()
        return Y
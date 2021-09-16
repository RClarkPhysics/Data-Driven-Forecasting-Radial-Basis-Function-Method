#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 16:04:52 2021

@author: randallclark
"""
import numpy as np
from numba import njit

class MultiQuad:
    """
    This is how Data Driven Forecasting is performed with a Multi Quadratic function is used as the Radial basis Function for
    interpolatio. All else remains the same. Ridge regression is used for training. The multi quadratic form is
    sqrt(||Data-Center||^2+R^2)
    inputs:
        Xdata - the data to be used for training and centers
        length - amount of time to train for
        p - used to choose centers. Basic choice: every p data points becomes a center
        beta - regularization term
        R - parameter used in the Radial basis Fucntion
        D - Dimension of th system being trained on
    """
    def FuncApproxF(self,Xdata,length,p,beta,R,D):
        self.D = D
        XdataT = Xdata.T
        #To Create the F(x) we will need only X, generate Y, give that to the Func Approxer
        #Xdata will need to be 1 point longer than length to make Y
        #Make sure both X and Y are in D by T format
        Ytarget = self.GenY(Xdata,length,D)
        
        #Here you choose how you will define your ceneters:
        #C = X  #Basic Bitch Centers
        NcLength = int(length/p)
        C = np.zeros((NcLength,D))
        for i in range(NcLength):
            C[i] = XdataT[i*p]
        
        #Make the X matrix in Ridge Regression now
        PhiMat = self.CreatePhiMat(XdataT,C,length,NcLength,R)
        
        #Perform RidgeRegression
        YPhi = np.zeros((D,NcLength))
        for d in range(D):
            YPhi[d] = np.matmul(Ytarget[d],PhiMat.T)
        PhiPhi = np.linalg.inv(np.matmul(PhiMat,PhiMat.T)+beta*np.identity(NcLength))
        W = np.zeros((D,NcLength)) 
        for d in range(D):
            W[d] = np.matmul(YPhi[d],PhiPhi)
        
        
        @njit
        def newfunc(x):
            f = np.zeros(D)
            for d in range(D):
                for c in range(NcLength):
                    f[d] = f[d] + W[d][c]*(np.linalg.norm(x-C[c])**2+R**2)**0.5
            return f
        
        self.FinalX = np.zeros(D)
        for d in range(D):
            self.FinalX[d] = Xdata[d][length-1]
        
        return newfunc
        
    
    
    
    """
    Predict ahead in time using the F(t)=dx/dt we just built
    input:
        F - This is the function created above, simply take the above functions output and put it into this input
        PreLength - choose how long you want to predict for
        Xstart - Choose where to start the prediction, the standard is to pick when the training period ends, but you can choose it
                    to be anywhere.
    """
    def PredictIntoTheFuture(self,F,PreLength,Xstart):
        @njit
        def makePre(D,FinalX):
            Prediction = np.zeros((PreLength,D))
            #Start by Forming the Bases
            Prediction[0] = F(FinalX)
            
            #Let it run forever now
            for t in range(1,PreLength):
                Prediction[t] = F(Prediction[t-1])
            return Prediction
        
        Prediction = makePre(self.D,Xstart)
        Prediction = Prediction.T
        return Prediction
        
        
        
    """
    These are secondary Functions used in the top function. You need not pay attention to these unless you wish to understand or alter
    the code.
    """ 
    def CreatePhiMat(self,X,C,Nlength,NcLength,R):
        @njit
        def getMat():
            Mat = np.zeros((NcLength,Nlength))
            for i in range(NcLength):
                for j in range(Nlength):
                    Mat[i][j] = (np.linalg.norm(X[j]-C[i])**2+R**2)**0.5
            return Mat
        Mat = getMat()
        return Mat    
        
    def GenY(self,Xdata,length,D):
        @njit
        def makeY():
            Y = np.zeros((D,length))
            for d in range(D):
                for t in range(length):
                    Y[d][t] = Xdata[d][t+1]
            return Y
        Y = makeY()
        return Y
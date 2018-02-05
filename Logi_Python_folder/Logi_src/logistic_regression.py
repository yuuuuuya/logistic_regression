import pandas as pd
import sys
sys.path.append('../Logi_data/')
sys.path.append('../Logi_src/')
import numpy as np
import matplotlib.pyplot as plt
import random
from sympy import*
import datautil as dutil
import funcutil as futil
import pdb


'''
Make Robins Monrroe function.

If update(loop) this function many times, we can find best parameter(a and W)

Input  : array(shape of input data), vector(shape of output data),
         parameter(vector(number of types of input data), parameter(scalor), number of types of data
         stepsize, number of loop, number of types of data
Output : vector(updated parameter(W)), scalor(update parameter(a)),
         scalor(updated value of Likelihood(yuudo) function)
'''

def robins_monroe(X, T, a, W, stepsize, loopnum, d=5, coeffi=0, penamodel="ridge"):
    recordW = np.zeros((loopnum,d))
    recordA = np.zeros((loopnum,1))
    Erecord = np.zeros(loopnum)

    valW = W#default of parameter W
    valA = a#default of parameter a
    Eval = futil.sigmoid_objective(X, T, valW, valA, coeffi, penamodel)#default of value of likelihood function
    recordW[0,:] = valW#record default of parameter W
    recordA[0] = valA#record default of parameter a
    Erecord[0] = Eval#record default of value of likelihood function

    for k in range(loopnum-1):

#To display of updates(value of likehoopd function) multiple of 10
        if np.mod(k,10)==0:
            print('%s iterations complete \t: Current Energy is %s'%(k,Erecord[k]))

#value of "Derivative with respect to W(a)" of likelihood(yuudo) function
        gradW = futil.diff_w_objective(X, T, valW, valA, d=5, coeffi=0, penamodel='ridge')
        gradA = futil.diff_a_objective(X,T,valW,valA)

#update parameters(W and a)
        updateW = gradW* stepsize
        updateA = gradA* stepsize
        valW = valW + updateW
        valA = valA + updateA
        recordW[k+1,:] = valW
        recordA[k+1] = valA
        Erecord[k+1] = futil.sigmoid_objective(X, T, valW, valA, coeffi, penamodel)

    return recordW, recordA , Erecord

'''
input  : inputdata(traindata), outputdata(traindata), iterration, stepSize ,
         dimention(number of kind of datas)
output : parameterW(vector) , parameterA(scalor)
These parameters are hight value of the Likelihood(yuudo) function.
'''
def trainModel(trainX, trainT, stepSize=1.0,loopnum=50, d=5, coeffi=0, penamode="ridge"):
    a = np.random.normal(0,1,size =1)
    W =np.random.normal(1, 10, size = d)
    Ws , As, Erecord = robins_monroe(trainX, trainT, a, W, stepSize, loopnum, d, coeffi, penamode)
    finalW = Ws[loopnum-1, :]
    finalA = As[loopnum-1]
    return finalW, finalA, Erecord

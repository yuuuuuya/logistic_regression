import pandas as pd
import sys
sys.path.append('../Logi_src/')
sys.path.append('../Logi_data/')

import numpy as np
import matplotlib.pyplot as plt
import random
from sympy import*
import imp
import pdb
import os


import datautil as dutil
import funcutil as futil


datapath = '../Logi_data/'

testfile = 'datatest.txt'
filepath = os.path.join(datapath, testfile )


'''
Goal is making functions that find rate accurate, falsepositive and falsenagative.
'''




'''
load parameters(W and a) finding by Robbins Monrroe

input  : paramerterpath, failename
Output : vector(paramerter(W)), scalor(paramerter(a))
'''
def para_load(parameterpath, filename = 'parameter_stepSize0.01.npy'):
    filepath = os.path.join(parameterpath, filename )
    sys.path.append('parameterpath')
    load_data = np.load(filepath)
    para_W = load_data[()]['W']
    para_A = load_data[()]['A']

    return para_W , para_A


'''
Load testdata function.
Input  : paramerterpath, failename
Output : input array(number of data, number of types of data), output vector(number of data,1)
'''
def load_testdata(testpath, filename = 'datatest.txt'):

    filepath = os.path.join(datapath, filename )
    sys.path.append('testpath')
    test1_df = pd.read_csv(filepath, sep = ',' )
    input_array = dutil.drop(test1_df)
    Norma_input = dutil.Normalization(input_array)

    output_array = dutil.target_val(test1_df)

    return Norma_input  , output_array


'''
Probability of occupancy = 1 in each of the data.
Input  : input array(number of data, number of types of data), vector(parameter(W)),scalor(parameter(a))
Output : vecor(p values. Probability of occupancy = 1 in each of the data)
'''
def computeP(test_input, paraW, paraA):
    W = paraW
    a = paraA
    X = test_input
    RowNum = len(X)
    #record = np.zeros(X.shape[0])
    P_valueRecord = np.zeros(RowNum)
    exponents = np.dot(X, W)+ a
    for i in range(RowNum):
        p_value = futil.sigmoid_function(exponents[i], 1)
        P_valueRecord[i] = p_value
    return P_valueRecord


'''
It distributes each of the probability to 1 or 0
The determination method is to ensure wether the plobability higher than threshold.
Input  : vector(plobabilityof occupancy = 1 in each of the data), threshold
Output : vector(elements is 1 or 0)
'''
def prediction(probabilities, threshold = 0.5):
    predicted = np.zeros(len(probabilities))
#extract number higher(lower) than threshold.
    posLoc = np.where(probabilities >= threshold)[0]
    negLoc =  np.where(probabilities < threshold)[0]
#replace elements to 0 or 1.
    predicted[posLoc] = np.ones(len(posLoc))
    predicted[negLoc] = np.zeros(len(negLoc))
    return predicted

'''
Research falsepositive ratio.
The way is to see ratio -1 that the result of difference.
Input  : vector(predicted), output of testdata
Output : scalor(falsepositive ratio)
'''
def FalsePositive(predicted, test_output):
    diff = test_output -  predicted
    sampleSize = float(len(predicted))
    fpLoc = np.where( diff < 0 )[0]
    fpNum = float(len(fpLoc))
    FPrate = fpNum / sampleSize

    return FPrate

'''
Research falsenegative ratio.
The way is to see ratio 1 that the result of difference.
Input  : vector(predicted), output of testdata
Output : scalor(falsenegative ratio)
'''
def FalseNegative(predicted, test_output):
    diff = test_output -  predicted
    sampleSize = float(len(predicted))
    fnLoc = np.where( diff > 0 )[0]
    fnNum = float(len(fnLoc))
    FNrate = fnNum / sampleSize

    return FNrate

'''
Research  persentage of correct answer.
The way is to see ratio 1-(falsepositive rotio plus falsenagative ratio)
Input  : vector(predicted), output of testdata
Output : scalor(persentage of correct answer)
'''
def Accurary(predicted, test_output):
     errRate = FalseNegative(predicted, test_output)+ \
     FalsePositive(predicted, test_output)
     return 1.-errRate

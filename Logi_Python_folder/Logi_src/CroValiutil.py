import numpy as np
import sys
sys.path.append('../Logi_src/')
sys.path.append('../Logi_data/')

import random
import pdb

from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold

import datautil as dutil
import logistic_regression as logi
import prediction as pre


'''
Make data for Cross Validation.

input  : input data(array), output data(vector), number of splits(scalor)
output : traindatas , testdatas

first dats of train_record and test_record is first cross validation datas.
...
third...                                      third...

'''


def CroVali_Data(X, T, splits):
    n_splits =  splits
    epochIdx = 0
    train_record = []
    test_record = []
    #cross validation(split array and loop)
    for train_idx, test_idx in StratifiedKFold(n_splits).split(X, T):
        xs_train = X[train_idx]
        t_train = T[train_idx]
        xs_test = X[test_idx]
        t_test = T[test_idx]
        #matched number of occupancy=0 with occupancy=1
        matchX, matchT = dutil.array_match(xs_train,t_train)

        traindatas = [matchX, matchT]
        testdatas = [xs_test, t_test]

        #Seving to a set matchX and matchT
        train_record = train_record + [traindatas]
        #Seving to a set xs_test and T_test
        test_record = test_record +[testdatas]

        epochIdx+=1

    return train_record, test_record


'''
solved p values in the form of a array.

input  : train_record, test_record
first dats of train_record and test_record is first cross validation datas.
...
third...                                      third...

output : P values in the type of array.
first low of P values array is p value of first cross validation.
'''

def Cross_P_values(train_record, test_record, stepSize, iter, d, coeffi, penamode):

    splitsNum = len(train_record)

    epochIdx = 0
    record_P_array = []

    for i in range(splitsNum):
        InputTrainX, InputTrainT = train_record[i]
        OutputTestX, OutputTestT = test_record[i]

        #finding best parameters to be hight Likelihood(yuudo) function.
        trainParaW, trainParaA, Erecord = logi.trainModel(InputTrainX, InputTrainT, stepSize, iter, d, coeffi, penamode)

        #Solve to p values in the form of a array
        P_valus_array = pre.computeP(OutputTestX, trainParaW, trainParaA)

        record_P_array = record_P_array + [P_valus_array]
        epochIdx+=1

    return record_P_array

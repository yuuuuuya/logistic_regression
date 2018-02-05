import numpy as np
import pdb

import sys
sys.path.append('../Logi_src/')

import funcutil as futil
import datautil as dutil

'''
def regularizeLaplace(rawdata):
    rawdataAve = np.mean(rawdata)
    rawdataStd = np.sqrt(np.var(rawdata))
    v = (rawdata - rawdataAve)/rawdataStd
    return v
'''

'''
In this function, to replace all output = 0 to output = -1
input  : outputdata
output : outputdata(only 1 or -1)
'''
def changeOutput(outputdata):
    T = outputdata
    zero_list = np.where(T ==0)[0]#extract number of T=0
    zeroNum = zero_list.shape[0]#number of T=0
    #to replace all T = 0 to T = -1()
    for i in range(zeroNum):
        num_0 = zero_list[i]
        T[num_0] = -1
    return T


'''
Make for feature vector.
This functin follow as sigmoid basis function.
input  : vector
output : feature vector follows as sigmoid basis function.
'''
def sigval(vec):
    vecNum = vec.shape[0]
    val = None
    for i in range(vecNum):
        val = futil.sigmoid_1(vec[i])
        vec[i] = val
    return vec

'''
Normalized vector making for gauss basis function.
input  : vector
output : normalized vector
'''
def regularizeGauss(rawdata):
    rawdataAve = np.mean(rawdata)
    rawdataStd = np.sqrt(np.var(rawdata))
    v = (rawdata - rawdataAve)**2/2*(rawdataStd)**2
    return v
'''
Make for feature vector.
This functin follow as gauus basis function.
input  : vector
output : feature vector follows as gauss basis function.
'''
def gaussval(vec):
    vecNum = vec.shape[0]
    val = None
    for i in range(vecNum):
        val = np.exp(-vec[i])
        vec[i] = val
    return vec
'''
in this function, to be feature array.
moreover, can select basis model(sigmoid or gauss)

input  : array , basis function model(sigmoid or gauss)
output : feature array (sigmoid or gauss model)
'''
#mode = 'Sigmoid' or 'Gauss'
def basis_funModel(array, mode):
    row, col = array.shape
    VecRecord = []
    for k in range(col):
        if mode == 'Sigmoid':
            vec = array[:,k]#extract kth vec of array.
            regularize_vec = dutil.regularizeLaplace(vec)#Normalize k row by regularizeLaplace()
            sig_vec = sigval(regularize_vec)#to be value of sigmoid.
            VecRecord += [sig_vec]#record.
        elif mode == 'Gauss':
            vec = array[:,k]
            regularize_vec = regularizeGauss(vec)#Normalize k row by regularizeGauss()
            gauss_vec = gaussval(regularize_vec)
            VecRecord += [gauss_vec]
        else:
            raise NotImplementedError

        #pdb.set_trace()
    norm_array = np.r_[VecRecord]#list combine for longwise(tate) direction.
    return np.transpose(norm_array)

'''
In this function, judge whether classified corrctly.
If classified corrctly, output to be 0.
If not calssified corrctly, output is corrent(imanomama) value.

input  : inputdata's vector(row of feature array), parameterW,
         outputdata's scalor(compatitive(taiou) inputdata's vector, value of outputdata is 1 or -1),
         Regularization coefficient(keisuu), normalization model
output : scaor (judged scalor)
'''
def Judgefunction(feature_vec, parameterW, output_scalor, coeff, penamodel):
    X = feature_vec
    W = parameterW
    t = output_scalor

    val = np.dot(X,W)
    penalty = futil.penaltyModel(W, coeff, penamodel)
    judge_val = (val +penalty)*t

    #pdb.set_trace()
    if judge_val > 0:
        judge_val = 0
    elif judge_val < 0:
        pass
    else:
        raise NotImplementedError

    return judge_val

'''
In this function, memory judged val in the form of vector.

input  : feature array, parameterW, outputdata (vector that elements are 1 or -1),
         Regularization coefficient(keisuu), normalization model
output : vector(elements are judged by Judgefunction)
'''

def judgedRecord(feature_array, parameterW, outputdata, coeff, penamodel):
    X = feature_array
    W = parameterW
    T = outputdata
    row ,col = X.shape

    recordVec = np.zeros(row)

    for i in range(row):
        judge_val = Judgefunction(X[i,:], parameterW, T[i], coeff, penamodel)
        recordVec[i] = judge_val

    return recordVec

'''
This function make a list that error classified number.
input  : judged vector. This elements are judged.
the way is if valuation of perseptron error function >0, judge is 0.
if valuation of perseptron error function <0, judge is pass(output same value)

output : number list that judge is pass.
'''

def errorNumlist(judgeRecord):
    errorNumlist = np.where(judgeRecord < 0)[0]
    return errorNumlist

'''
In this function, make a weight val for adding parameter W.
input  : feature vector that not judged 0, parameterW,
         output scalor that compatitive(taiou) feature vector.
         Regularization coefficient(keisuu), normalization model
output : weight val be made feature vector and output scalor compatitive feature vector.
'''

def weightfunction(feature_vec, parameterW, output_sca, coeff, penamodel):
    X = feature_vec
    W = parameterW
    t = output_sca

    penalty = futil.penaltyModel(W, coeff, penamodel)
    weightval = (X + penalty)*t

    return weightval


'''
This function make for checking whether error function decrease.

input  : feature array, parameterW, outputdata (vector that elements are 1 or -1),
         Regularization coefficient(keisuu), normalization model
output : vector(elements are judged by Judgefunction)
'''
def PersepErrorFun(feature_array, parameterW, outputdata, coeff, penamodel):
    X = feature_array
    T = outputdata
    W = parameterW

    row, col = X.shape
    val = None

    for i in range(row):
        val =+ val + Judgefunction(X[:,i], parameterW, T[i], coeff, penamodel)
    return val










'''
If x is a or more, output is 1.
If x is less than a, output is -1.

input  : x and a are scalor
output :1 or -1
'''

def stepfunciton(x,a):
    val = None
    if x >= a :
        val = 1
    elif x < a :
        val = -1
    else:
        raise NotImplementedError
    return val

def stepfun_vec(vec,a):
    vecNum = vec.shape[0]
    val = None
    for i in range(vecNum):
        val = stepfunciton(vec[i],a)
        vec[i] = val
    return vec



'''
array to be vector.
To extract rows and to conbine the rows.

input  : array
output : vector(conbining rows)
'''
def vectorization(array):
    RowNum, ColNum = array.shape
    ElementNum = (RowNum)*(ColNum)

    VecRecord = []

    for i in range(RowNum):
        VecRecord = VecRecord + [array[i,:]]

    objecVec = np.concatenate(VecRecord ,axis =0 )#conbine for crosswise direction

    return objecVec

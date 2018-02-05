import pandas as pd
import sys
sys.path.append('../Logi_data/')
sys.path.append('../Logi_src/')

import numpy as np
from numpy.random import*
import matplotlib.pyplot as plt

from sympy import*

import pdb

import datautil as dutil



'''
This util go toward making function for finding best parameter(a and W) by Robbins monrroe.
'''

'''
This sigmoid_0(),sigmoid_1() functions make for avoiding error "overflow encountered in exp"
get an error in -709 the following valuse(-709ikano ataideha era-gaderu)
http://hamukazu.com/2015/07/31/mathematical-derivation-in-numerical-computation/
'''

def sigmoid_0(x):
    val = None
    if x< -709:
        val = 0.0
    elif x>= -709:
        val =  1./(1. + np.exp(-x))
    else:
        NotImplementedError
    return val

def sigmoid_1(x):
    val = None
    if x< -709:
        val = 1.0
    elif x>= -709:
        val =  np.exp(-x)/(1. + np.exp(-x))
    else:
        NotImplementedError
    return val
'''
sigmoid function
If output is 1, select exp(-z)/(1. + exp(-z))
If output is 0, selest 1./(1. + exp(-z))

input  : scalor, scalor(output data)
output : scalor
'''
def sigmoid_function(z, T):
    val = None
    if T == 0:
        val= sigmoid_0(z)
    elif T == 1:
        val= sigmoid_1(z)
    else:
        raise NotImplementedError
    return val


'''
Normalization model (ridge regression and lasso regression)
input  : vector(parameter)
output : scalor(one of the penlty. )
'''
def ridge(W):
    penalty = np.dot(W,W)#penalty is squre(jijou) norm of prameter W
    return penalty

def lasso(W):
    factorial = np.dot(W,W)#factorial is kaijou
    penalty = np.sqrt(factorial)#penalty is norm of poarameter W
    return penalty

'''
penalty model(penalty is ridge regression or lasso regression)
input  : vector(parameter), Regularization coefficient(keisuu), normalization model
output : scalor(penalty for ordering to nomalize)
'''

def penaltyModel(W, coeffi, penamodel):
    if penamodel=="ridge":
        val = ridge(W)
    elif penamodel =="lasso":
        val = lasso(W)
    else:
        raise NotImplementedError
    penalty = (val)*(coeffi)
    return penalty

'''
This function can select whether normalize or not.
Moreover, can select nomalize models 'redge or lasso'
input  : vector (row of the input data),scalor(row of the output data), vector(parameter)
         scalor(parameter), Regularization coefficient(keisuu), normalization model
output : scalor(normalized value or not normalized value)
'''
def sigmoidModel(X, T, W, a, coeffi, penamodel):#coefficient:係数
    exponent = np.dot(W, X) + a
    penalty = penaltyModel(W,coeffi,penamodel)
    regu_para = exponent + penalty
    sig_val = sigmoid_function(regu_para, T)
    return sig_val


'''
This function is Likelihood(yuudo) function.
Input  : array(input data), vector(output data),vector(parameter), scalor(parameter)
         Regularization coefficient(keisuu), normalization model
Output : scalor
'''

def sigmoid_objective(X, T, W, a, coeffi, penamodel):
    record = np.zeros(T.shape[0])
    val = 0
    for k in range(T.shape[0]):
        val = val + sigmoidModel(X[k],T[k],W,a,coeffi,penamodel)
        record[k] = val
    return record[T.shape[0]-1]



'''
Make "Derivative with respect to W" of likelihood(yuudo) function.
'''

'''
This diff_sigmoid_0(),diff_sigmoid_1() functions make for avoiding error "overflow encountered in exp"
get an error in -709 the following valuse(-709ikano ataideha era-gaderu)
http://hamukazu.com/2015/07/31/mathematical-derivation-in-numerical-computation/
'''

def diff_sigmoid_0(x):
    val = None
    if x< -354:
        val = 0.0
    elif x>= -709:
        val = np.exp(-x)/(1. + np.exp(-x))**2
    else:
        NotImplementedError
    return val

def diff_sigmoid_1(x):
    val = None
    if x< -354:
        val = 1.0
    elif x>= -709:
        val = -np.exp(-x)/(1. + np.exp(-x))**2
    else:
        NotImplementedError
    return val

'''
Make derivative of sigmoid function.

input  : z:scalor , T:scalor(one of output data)
output : scalor
'''
def sigmoid_derivative(z, T):
    val = None
    if T ==0:
        val = diff_sigmoid_0(z)
    elif T ==1:
        val = diff_sigmoid_1(z)
    else:
        raise NotImplementedError
    return val

'''
derivative funstion of Normalize model(redge regression or lasso regression)
input  : vector(parameter)
output : vector(derivative value)
ridge_diff output is 2*W(parameter). lasso_diff output's element is if wi > 0 => 1, if wi < 0 => -1
'''

def ridge_diff(W):
    penaltyGrad = 2*W
    return penaltyGrad

def lasso_diff(W):
    penaltyGrad = np.sign(W)
    return penaltyGrad
    '''
    elementNum= W.shape[0]
    record = np.zeros(elementNum)
    for i in range(elementNum):
        if W[i]>0:
            record[i]=1
        elif W[i]<0:
            record[i]=-1
        else:
            record[i]= None
    return record
    '''

'''
In this function, can select whether normalized derivative value or not normalized.
Normalized derivative value is derivative redge or lasso.
input  : scalor(parameter), Regularization coefficient, normalization model
output : penalty derivated with respect to W(diff value times coefficient)
'''
def diff_penaltyModel(W, coeffi, penamodel):
    if penamodel =="ridge":
        diff = ridge_diff(W)
    elif penamodel =="lasso":
        diff = lasso_diff(W)
    else:
        raise NotImplementedError

    diff_penalty = (diff)*(coeffi)
    return diff_penalty


'''
"Derivative with respect to W" function.
But this function is not "Derivative with respect to W" of likelihood(yuudo) function.

Input  : vector(row of input data), scalor(one of output data),
         vectorparameter(parameter), scalorparameter(parameter)
         Regularization coefficient, normalization model
output : vector(differential value times(kakeru) vector(row of input data))
'''
def sigmoid_derivative_W(X, T, W, a, coeffi, penamodel):
    penalty = penaltyModel(W, coeffi, penamodel)
    exponent = np.dot(W, X) + a + penalty
    #A is scalor
    A = sigmoid_derivative(exponent,T)

    penalty_diff = diff_penaltyModel(W, coeffi, penamodel)
    #B is array
    B = X + penalty_diff

    return A*B


'''
This function make for confirming whether differential value with respect to W is right.

Input  : vector(row of input data), scalor(one of output data), vector(parameter),
         scalor(parameter) , h:small value , i:location of elements(low:0 , high:4)
         Regularization coefficient, normalization model
Output : scalor(derivated i-th element)
'''
def check_diff_W(X, T, W, a, h, i, coeffi, penamodel):
    A = i_matrix(i)
    F = (sigmoidModel(X,T,W+A*h,a,coeffi,penamodel) - sigmoidModel(X,T,W,a,coeffi,penamodel))/h
    return F
'''
This function make that (i.j) componentes(seibunn) is 1
i:location of elements(low:0 , high:4)
'''
def i_matrix(i):
    A = np.zeros(5)
    A[i] = 1
    return A

'''
"Derivative with respect to W" of likelihood(yuudo) function.
Input  : array(shape of input data), vector(shape of output data),
         parameter(vector(number of types of input data), parameter(scalor) , number of types of data
Output : scalor
'''

def diff_w_objective(X, T, W, a, d, coeffi, penamodel):
    record = np.zeros((T.shape[0],d))
    val = [0]*d
    for i in range(T.shape[0]):
        val += sigmoid_derivative_W(X[i], T[i], W, a, coeffi, penamodel)
        #check_diff_W(X[i], T[i], W, a, h=0.00001, i=3, coeffi=0, penamodel='ridge')
        record[i,:] = val
    return record[T.shape[0]-1,:]





'''
Make "Derivative with respect to a " of likelihood(yuudo) function.
'''

'''
"Derivative with respect to a" function.
But this function is not "Derivative with respect to a" of likelihood(yuudo) function.

Input  : vector(row of input data), scalor(one of output data),
         parameter(vector(number of types of input data), parameter(scalor)
output : scalor
'''
def sigmoid_derivative_a(X, T, W, a):
    exponent = np.dot(W, X) + a
    F = sigmoid_derivative(exponent,T)
    return F

'''
This function make for confirming whether differential value with respect to a is right.

Input  : vector(row of input data), scalor(one of output data), parameter(vector(number of types of input data),
         parameter(scalor) , h:small value
output : scalor
'''
def check_diff_a(X, T, W, a, h, coeffi, penamodel):
    F =( sigmoidModel(X, T, W, a+h, coeffi, penamodel)- sigmoidModel(X, T, W, a,coeffi, penamodel))/h
    return F


'''
"Derivative with respect to a" of likelihood(yuudo) function.
Input  : array(shape of input data), vector(shape of output data),
         parameter(vector(number of types of input data), parameter(scalor) , number of types of data
Output : scalor
'''
def diff_a_objective(X,T,W,a):
    record = np.zeros(T.shape[0])
    val =0
    for i in range(T.shape[0]):
        val = val + sigmoid_derivative_a(X[i], T[i], W, a)
        record[i] = val
    return record[T.shape[0]-1]

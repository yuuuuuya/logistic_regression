import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
from sympy import*
import imp
import pdb
import os

from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
#http://segafreder.hatenablog.com/entry/2016/10/18/163925


sys.path.append('../Logi_src/')
sys.path.append('../Logi_data/')


datapath = '../Logi_data/'

trainfile = 'datatraining.txt'
filepath = os.path.join(datapath, trainfile )


testfile = 'datatest.txt'
test_filepath = os.path.join(datapath, testfile )

test2file = 'datatest2.txt'
test2_filepath = os.path.join(datapath, test2file )


'''
This util go toward(mokuhyou) making X(iput data) and T(outputdata).
'''

'''
Mutch the number of occupancy=0 with occupancy=1.
When pick up data, must select at random.
'''

'''
Shuffle the index of dataframe.
Input  : dataframe
Output : Shuffled rows of dataframe
'''
def Df_shuffle(df):
    index = list(df.index)
    random.shuffle(index)
    df = df.ix[index]
    df.reset_index()
    return df

'''
Input  : dataframe
Output : dataframe
The output dataframe mutched number of occupancy=0 with occupancy=1 and shuffled the data number.
'''

def Df_match(df):
    df0 = df[df.Occupancy !=0]
    df1 = df[df.Occupancy !=1]
    shuf_df0 = Df_shuffle(df0)
    shuf_df1 = Df_shuffle(df1)
    df0_num = shuf_df0[shuf_df0.Occupancy !=0].shape[0]
    df1_num = shuf_df1[shuf_df1.Occupancy !=1].shape[0]
    if df0_num >=df1_num:
        new_df0 = shuf_df0[0:df1_num]
        new_df = pd.concat([new_df0 , shuf_df1])
    elif df0_num <df1_num:
        new_df1 = shuf_df1[0:df0_num]
        new_df = pd.concat([shuf_df0 , new_df1])
    else :
        new_df = pd.concat([shuf_df0, shuf_df1])
    return new_df





'''
Make  X(iput data)
'''

'''
Droped Occupation and date for extracting necessary dataes.
(Necessary dataes are Temperature,Humidity,Light,CO2, and HumidityRatio.)
Input  : dataframe
Output : array(number of data, number of types of data)
'''
def drop(df):
    drop_df =  df.drop(['date' , 'Occupancy'] , axis=1)
    drop_array = drop_df.as_matrix()
    return drop_array


'''
Normalize X(input dada)
Input  : array(vector)
Output : array(normalized vector)
'''
#make Normalized array of max is 1 and min is 0 by odividing norm
def regularizeNorm(rawdata):
    v = rawdata/np.linalg.norm(rawdata)
    return v

#Appropriate(teiktou) normalization
def regularizeteMax(rawdata):
    rawdataMax = np.max(rawdata)
    rawdataMin = np.min(rawdata)
    rawdataAve = np.mean(rawdata)
    rawdataDelta = rawdataMax - rawdataMin
    v = (rawdata - rawdataAve)/rawdataDelta
    return v

#make Normalized array of mean is 0 and variance is 1
def regularizeLaplace(rawdata):
    rawdataAve = np.mean(rawdata)
    rawdataStd = np.sqrt(np.var(rawdata))
    v = (rawdata - rawdataAve)/rawdataStd
    return v


'''
input  : array(number of data, number of types of data)
output : normalized array(number of data, number of types of data)
'''
def Normalization(array, mode = 'Max'):
    row, col = array.shape
    mylist = []
    for k in range(col):
        if mode == 'Max':
            mylist += [regularizeMax(array[:,k])]#Normalize k row by regularizeMax()
        elif mode == 'Norm':
            mylist += [regularizeNorm(array[:,k])]#Normalize k row by regularizeNorm()
        elif mode == 'Laplace':
            mylist += [regularizeLaplace(array[:,k])]#Normalize k row by regularizeLaplace()
        else:
            raise NotImplementedError
        #pdb.set_trace()

    norm_array = np.c_[mylist]#np.c_() get rows combine
    return np.transpose(norm_array)




'''
Make  T(output data)
'''

'''
Extract data of Occupancy.
Input : dataframe
Output : array(number of data,1)
'''
def target_val(df):
    ext_df = df['Occupancy']#extract data of Occupancy
    ext_array = ext_df.as_matrix()
    return ext_array



'''
This function generate X(input data) and T(output data) from 'path', 'filename' and 'Nomalizatioon modele'
'''
'''
Input  : path , filename , Nomalization modele
Output : input data(array(number of data, number of types of data)) , output data(array(number of data,1))
'''

#datapath='../Logi_data/'
def process_traindata(datapath, filename = 'datatraining.txt', mode = 'Laplace'):
    sys.path.append('path')
    train_df = load_traindata(datapath, filename)
    match_df = Df_match(train_df)#match the number of T=0 with T=1
    input_array = drop(match_df)#extract necessary data for maiking input data
    norm_input = Normalization(input_array, mode = mode)#normalize data for making input data

    output_array = target_val(match_df)#extract occupancy data for making outputdata

    return norm_input, output_array



'''
load traindata,testdata and test2data.
Input  : datapath, failename
Output : dataflame

First function is uploading function of traindata.
Second function is uploading function of testdata.
Third function is uploading unction of testdata1.
'''
def load_traindata(path, filename = 'datatraining.txt'):
    sys.path.append('path')
    filepath = os.path.join(path, filename )
    df = pd.read_csv(filepath, sep = ',' )
    return df

def load_testdata(path, filename = 'datatest.txt'):
    sys.path.append('path')
    filepath = os.path.join(path, filename )
    df = pd.read_csv(filepath, sep = ',' )
    return df

def load_testdata(path, filename = 'datatest2.txt'):
    sys.path.append('path')
    filepath = os.path.join(path, filename )
    df = pd.read_csv(filepath, sep = ',' )
    return df

'''
Make original data.
It is way to match traindata, testdata and test2data.
Input  : datapath
Output : dataflame
'''

def combineData(path,trainfilename='datatraining.txt', testfilename='datatest.txt', test2filename='datatest2.txt'):
    sys.path.append('path')
    traindf = load_traindata(path, trainfilename)
    testdf = load_traindata(path, testfilename)
    trest2df = load_traindata(path, testfilename)
    combine_df = pd.concat([traindf, testdf, trest2df])#combine traindf, testdf and trest2df
    oridf = Df_shuffle(combine_df)#shuffle
    return oridf

'''
Extract inputdata and outputdata from origindata.
Input  : orijin dataframe , normalize type
Output : array(inputdata of origindata) , array(output of orijindata)
'''

def load_originaldata(oridf, mode = 'Laplace'):
    input_array = drop(oridf)#drop data and occupancy
    input = Normalization(input_array, mode = mode)#normalize

    output = target_val(oridf)#extract occupoancy

    return input, output

'''
input  : array
output : array shuffled rows
'''
def array_shuffle(array):
    #行列の行をシャフルする。
    #参考URL：https://stackoverflow.com/questions/35646908/numpy-shuffle-multidimensional-array-by-row-only-keep-column-order-unchanged
    shuffled_array = np.take(array,np.random.permutation(array.shape[0]),axis=0)
    return shuffled_array

'''
input  : inputdata(array), outputdata(vector)
output : array shuffled rows and matched number of occupancy=0 with occupancy=1
'''
def array_match(xs_train,ts_train):

    trainArray = np.c_[xs_train,ts_train]#trainデータの結合
    #今、５行目がoutputdata。そのため、５行目が、１のdataがoccupancy=1のdata。
    occu_1 = trainArray[trainArray[:,5]==1]#occupancy=1のdataを抽出。
    occu_0 = trainArray[trainArray[:,5]==0]#occupancy=0のdataを抽出。
    num_1 = occu_1.shape[0]#occupancy=1のdata数
    num_0 = occu_0.shape[0]#occupancy=0のdata数
#多い方のデータからは、ランダムにデータを抽出したいため、shuffleする。
    shuffle_1 = array_shuffle(occu_1)
    shuffle_0 = array_shuffle(occu_0)

    if num_1 >=num_0:
        new_1 = shuffle_1[0:num_0,:]#0~occupancy=0のdata数まで切り取る。
        new_array = np.r_[new_1 , shuffle_0]#縦方向に行列を結合する
    elif num_1 <num_0:
        new_0 = shuffle_0 [0:num_1,:]
        new_array = np.r_[shuffle_1 , new_0]
    else :
        new_array = np.r_[shuffle_1 , shuffle_0]

    match_input =  new_array[:,0:5]#trainデータをinputとoutoputに分割。
    match_output = new_array[:,5]

    return match_input, match_output

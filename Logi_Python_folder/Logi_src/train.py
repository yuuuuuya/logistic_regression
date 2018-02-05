#-*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../Logi_src/')
sys.path.append('../Logi_data/')
import argparse
import os
import logistic_regression as Logi
import datautil as dutil
import time
import random
import pdb

datapath = '../Logi_data/'


'''
Excute Robbins Monrroe for finding best parameters(W and a) in this place.
The code make for executing on terminal.
'''

def main(args):

    print("Initiating the training sequence...")
    #load teraining data.
    X, T= dutil.process_traindata(datapath, filename = args.trainFile, mode = args.normMode)
    #pdb.set_trace()

    start = time.time()
    #number of loop
    iter = args.iteration
    d = args.dimension
    coeffi = args.coefficient
    penamode = args.penaMode

    #Initial values
    a = np.random.normal(0,1,size =1)
    W =np.random.normal(1, 10, size = 5)
    Ws , As, Energy = Logi.robins_monroe(X, T, a, W, args.stepsize, iter, d, coeffi, penamode)


    parname = args.parname+'_stepSize%s_iter%s_normMode%s'%(args.stepsize, iter, args.normMode)
    figname = args.figname+'_stepSize%s_iter%s_normMode%s'%(args.stepsize, iter, args.normMode)


    figpath = os.path.join(datapath, figname + '.png' )
    parameterNamePath = os.path.join(datapath, parname + '.npy')

    end = time.time()
    print(u"演算所要時間%s秒"%(end-start))
    print(u"parameter saved at %s"%parname)

    #load uploaded parametr and valuse of likelihood function.
    finalW = Ws[iter-1, :]
    finalA = As[iter-1]
    #save of type of directory
    finalParameters = {}
    finalParameters['A']= finalA
    finalParameters['W']= finalW
    #plot upload values of likelihood function and save graph
    fig = plt.figure(figsize = (20,20) )
    plt.plot(Energy)
    plt.title("Energy Transition")
    plt.xlabel("iterations")
    plt.savefig(figpath)


    #save finalparameters(W and a) on paramternamepath
    np.save(parameterNamePath, finalParameters)





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Logistic regression')

    parser.add_argument('--figname', '-fig', type =str, default = 'trajectory' )
    parser.add_argument('--parname', '-par', type =str, default = 'parameter' )
    parser.add_argument('--iteration', '-i', type =int, default = 50 )
    parser.add_argument('--dimension', '-d', type =int, default = 5 )
    parser.add_argument('--stepsize', '-s', type =float, default = 0.1  )
    parser.add_argument('--trainFile', '-f', type =str, default = 'datatraining.txt' )
    #normMode:'Max','Norm','Laplace'
    parser.add_argument('--normMode', '-m', type = str, default = 'Laplace'  )
    parser.add_argument('--coefficient', '-coe', type = float, default = 0  )
    #'redge','lasso'
    parser.add_argument('--penaMode', '-pena', type = str, default = 'ridge' )


    args = parser.parse_args()

    main(args)
    os.system('say "ロジスティックモデル学習終了"')
    os.system('open -a Finder %s'%datapath)

#-*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../Logi_src/')
sys.path.append('../Logi_data/')
import argparse
import os

import prediction as pre

datapath = '../Logi_data/'

parameterfile = 'parameter.npy'
filepath = os.path.join(datapath, parameterfile)

'''
See the persentage of correct answer, falsepositive and falsenagative.
The code make for executing on terminal.
'''



def main(args):
    #generate array(11 elements from 0 to 1:[0.,0.1,0.2,...,1.0])
    parameterGrid = np.linspace(0,1, args.gridnumber +1)
    acc_record = np.zeros(args.gridnumber +1)
    fn_record = np.zeros(args.gridnumber +1)
    fp_record = np.zeros(args.gridnumber +1)

    para_W , para_A = pre.para_load('../Logi_data/', filename = 'parameter_stepSize%s_iter%s_normMode%s.npy'%(args.stepsize, args.iter, args.normMode) )
    test_X , test_T = pre.load_testdata('../Logi_data/', filename = 'datatest.txt')


    probabilities = pre.computeP(test_X, para_W, para_A)

    #Run the prediction sequence for each threshold
    for i in range(args.gridnumber +1):
        predicted = pre.prediction(probabilities ,  threshold = parameterGrid[i])
        fnRate= pre.FalseNegative(predicted, test_T)
        fpRate= pre.FalsePositive(predicted, test_T)
        acc  = pre.Accurary(predicted, test_T)

        acc_record[i] = acc
        fn_record[i] = fnRate
        fp_record[i] = fpRate


    maxAcc = np.max(acc_record)
    bestThreshold = parameterGrid[np.where(acc_record == maxAcc)[0][0]]

    print('Best accuracy achieved at threshold=%s with accuracy %s'%(bestThreshold, maxAcc))

    figname = args.figname+'_stepSize%s_normMode%s_macAcc%s'%(args.stepsize, args.normMode, maxAcc)


    figpath = os.path.join(datapath, figname + '.png' )


    fig = plt.figure(figsize = (20,20) )
    plt.plot(acc_record, color='red', label='ratio of accurate')
    plt.plot(fp_record, color='green', label='ratio of falsepositive')
    plt.plot(fn_record, color='blue', label='ratio of falsenegative')
    plt.legend()
    plt.title("Persentage of correct answer, False positive and False negative")
    plt.xlabel("threshold")
    plt.savefig(figpath)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Logistic regression')

    parser.add_argument('--figname', '-fig', type =str, default = 'test' )
    parser.add_argument('--stepsize', '-s', type =float, default = 1.  )
    parser.add_argument('--gridnumber', '-g', type = int, default = 100  )
    parser.add_argument('--iter', '-i', type = int, default = 100  )
    #normMode:'Max','Norm','Laplace'
    parser.add_argument('--normMode', '-m', type = str, default = 'Laplace'  )


    args = parser.parse_args()

    main(args)
    os.system('say "ヴァリデーションテスト終了"')
    os.system('open -a Finder %s'%datapath)

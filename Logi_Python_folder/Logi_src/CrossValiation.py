#import pandas as pd
import sys
sys.path.append('../Logi_src/')
sys.path.append('../Logi_data/')
import numpy as np
import matplotlib.pyplot as plt
#import random
import pdb
import os
import argparse
import time

import datautil as dutil
import logistic_regression as logistic
import prediction as pre
import CroValiutil as cross

from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
#http://segafreder.hatenablog.com/entry/2016/10/18/163925

datapath = '../Logi_data/'




def main(args):

    start = time.time()


    iter = args.iteration
    stepSize = args.stepsize
    d=args.dimension
    penamode = args.penaMode
    Tgridnum = args.Tgridnumber
    Lgridnum = args.Lgridnumber
    splitsNum = args.splits


    fig = plt.figure(figsize = (20, 20))

    thre_parameterGrid = np.linspace(0,1, Tgridnum +1)
    coe_parameterGrid = np.linspace(0.0,1.0, Lgridnum +1)
    Accu_record_array = np.zeros([Lgridnum+1,Tgridnum+1])

    figname = args.figname+'_penaltyModel%s'%(penamode)

    #load original data frame.
    oridf = dutil.combineData('../Logi_data/',trainfilename='datatraining.txt', testfilename='datatest.txt', test2filename='datatest2.txt')
    #load normalised inputdata and outputdata.
    X,T = dutil.load_originaldata(oridf, mode = args.normMode)



    #first row of train_record and test_record are first data(array) of cross validation.
    #number of row is number of split
    train_record, test_record = cross.CroVali_Data(X, T, splitsNum)


    print('Start Cross Validation')# (threshold:%s,coefficient:%s)'%(threshold,coeffi)




    for k in range(Lgridnum + 1):

        coeffi=coe_parameterGrid[k]

        epochIdx_P_vector = 0
        record_P_vector = []

        epochIdx_test = 0
        testDataRecord = []

        for j in range(splitsNum):
            InputTrainX, InputTrainT = train_record[j]
            InputTestX, OutputTestT = test_record[j]
            print('%s th training'%j)
            #finding best parameters to be hight Likelihood(yuudo) function.
            trainParaW, trainParaA, Erecord = logistic.trainModel(InputTrainX, InputTrainT, stepSize, iter, d, coeffi, penamode)
            plt.plot(Erecord, label= 'lambda =%s, %sth cross validation'%(coeffi,j))

            #Solve to p values in the form of a array
            p_value_vector = pre.computeP(InputTestX, trainParaW, trainParaA)

            record_P_vector  = record_P_vector + [p_value_vector]
            epochIdx_P_vector  += 1

            testDataRecord = testDataRecord +[OutputTestT]
            epochIdx_test += 1

        #finding accuracy of some thresholds
        for i in range(Tgridnum + 1):
            threshold = thre_parameterGrid[i]

            epochIdx_accu = 0
            accuracyRecord = []

            for t in range(splitsNum):
                p_value = record_P_vector[t]
                testdata = testDataRecord[t]

                #distributes each of the probability to 1 or 0 by ensuring wether the plobability higher than threshold.
                predicted_array = pre.prediction(p_value_vector, threshold)

                accuracy = pre.Accurary(predicted_array, OutputTestT)

                print('%s th cross validation accuracy:%s'%(t,accuracy))

                #memory several pasentage of correct answers
                accuracyRecord = accuracyRecord + [accuracy]
                epochIdx_accu += 1


            #average of pasentage of correct answers
            average = np.sum(accuracyRecord)/splitsNum
            Accu_record_array[(Lgridnum)-k,i] = average

            print('accuracy average:%s, threshold:%s,coefficient:%s'%(average,threshold,coeffi))




    end = time.time()
    print("Penalty mode: %s, Data normalization mode:%s"%(penamode, args.normMode))
    print('Grid Search sequence completed.')
    print(u"演算所要時間%s秒"%(end-start))


    accuMax = np.amax(Accu_record_array)
    Row, bestCol = np.where(Accu_record_array== accuMax)

    #when recording Accu_record_array, subsituated to order from the bottom.
    #gyouha,sitakara junnbannni dainyusita.
    bestRow = (Lgridnum)-Row
    bestCoef = coe_parameterGrid[bestRow][0]
    bestThre = thre_parameterGrid[bestCol][0]
    print(accuMax)

    Title = "best threshold is: %s,  best reg Coeff is: %s"%(bestThre, bestCoef)
    print(Title)

    figpathHistory = os.path.join(datapath, figname + 'history.png')
    plt.legend()
    plt.title(Title)
    plt.savefig(figpathHistory)
    plt.close()


    figpath = os.path.join(datapath, figname + '.png' )



    plt.imshow(Accu_record_array,interpolation ="None")
    plt.title("Cross Validation Normalize Model:%s"%(penamode))
    plt.xlabel("Threshold")
    plt.ylabel("Normalization Coefficient")
    plt.savefig(figpath)




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Cross Valisation')


    parser.add_argument('--figname', '-fig', type =str, default = 'CrossValidation' )
    parser.add_argument('--iteration', '-i', type =int, default = 100 )
    parser.add_argument('--dimension', '-d', type =int, default = 5 )
    parser.add_argument('--stepsize', '-s', type =float, default = 0.001 )
    #normMode:'Max','Norm','Laplace'
    parser.add_argument('--normMode', '-m', type = str, default = 'Laplace' )
    parser.add_argument('--splits', '-spl', type = int, default = 10 )
    parser.add_argument('--Tgridnumber', '-tg', type = int, default = 20, help= 'Threshold grid number' )
    parser.add_argument('--Lgridnumber', '-lg', type = int, default = 20, help= 'lamb grid number'  )

    #'redge','lasso'
    parser.add_argument('--penaMode', '-p', type = str, default = 'ridge' ,help= 'regularization mode')


    args = parser.parse_args()

    main(args)
    os.system('say "分割交差検証終了"')
    os.system('open -a Finder %s'%datapath)

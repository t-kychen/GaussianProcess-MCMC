'''
Created on May 30, 2015

@author: K.Y. Chen
'''
import sys
import dataSet
import user
import framework
import numpy as np
import random

def mainExtract(data, col_idx, opt):
    '''
    same function as extractData() in dataSet class
    '''
    new_data = []
    
    if opt == "row":
        data = data.T
    
    for idx in col_idx:
        new_data.append(data[:,idx])
    new_data = np.asarray(new_data)
    
    if np.shape(new_data)[0] <= np.shape(new_data)[1]:
        new_data = new_data.T
    
    return new_data

def mainShuffle(data, tms):
    '''
    Shuffle data to prevent from overfitting
    :param data: ndarray, input data with size N*D
    :param tms: int, numbers of shuttling

    :return data: ndarray, data after shuttling
    '''
    t = 0
        
    while t < tms:
        random.seed(124)    # set random seed for reproduction
        randIdx = random.sample(range(0, np.shape(data)[0]), np.shape(data)[0])
        data = data[randIdx, :]
        t += 1
    
    return data

def removeCSZero(data):
    '''
    remove 0's in Condition Score
    
    @param data: original data to be processed
    '''
    cs = data[:,0]
    zeroRow = []
    for r in range(np.shape(cs)[0]):
        if cs[r] == 0:
            zeroRow.append(r)
    return np.delete(data, zeroRow, 0)

if __name__ == "__main__":

    numIters = int(sys.argv[1][2:])
    experiment = sys.argv[2][2:]        # single / cross / ar

    usr = user.UserInput()
    usr.setInput(region="good", gapMin=1, gapMax=7)

    whole = None
    for district in usr.inputDist:
        for year in usr.inputYr:
            d = dataSet.DataSet()
            d.getData(district, year)
            
            if len(usr.inputRoute) != 0:
                d.getRoute(usr.inputRoute)
            sect = d.data
            
            if whole is None:
                whole = sect
            else:
                whole = np.vstack((whole, sect))

    # column information
    # first column is data year whole[:,0]
    # distScore = whole[:,6]
    # condScore = whole[:,7]
    # rideScore = whole[:,8]
    # TRM: whole[:, 2:6]

    colExtracted = [7, 2, 3]
    whole = mainExtract(whole, colExtracted, "col")
    colName = np.reshape(d.feature, (1, len(d.feature)))
    colName = mainExtract(colName, colExtracted, "col")

    # adding TRM for each section
    trm = whole[:, 1]+whole[:, 2]
    trm = np.reshape(trm, (trm.shape[0], 1))
    whole = np.hstack((whole, trm))
    colName = np.append(colName, ["TRM"])

    colNotUsed = [1, 2]
    whole = np.delete(whole, colNotUsed, 1)
    colName = np.delete(colName, colNotUsed, 0)

    # divide data into good & bad condition regions @ TRM-60
    start = list(whole[:,1]).index(55)
    end   = list(whole[:,1]).index(165)
    if usr.inputReg == "bad":
        print "[info] focusing on BAD region..."
        whole = whole[:start, :]
    elif usr.inputReg == "good":
        print "[info] focusing on GOOD region..."
        whole = whole[start:end, :]
    
    # remove 0's from CS
    if colName[0] == "CONDITION_SCORE":
        whole = removeCSZero(whole)

    x = whole[:, 1:]
    y = np.reshape(whole[:, 0], (np.shape(whole)[0], 1))

    if experiment == "single":
        firstExp = framework.singleRun(data=whole)
        firstExp.execute(updOpt='mcmcSml', iterMCMC=numIters)

    elif experiment == "cross":
        secondExp = framework.crossValid(data=whole, window=4, gapArray=usr.inputGap)
        secondExp.execute(updOpt='mcmcSml', iterMCMC=numIters)

    elif experiment == 'ar':
        baseline = framework.autoregressive(data=whole, window=4, gapArray=usr.inputGap, lag=1)
        baseline.execute()

    else:
        print '[error] no such experiment available!'


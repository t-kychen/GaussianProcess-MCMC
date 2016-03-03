'''
Created on May 30, 2015

@author: Thomas
'''
import dataSet
import user
import framework
import numpy as np
import random
import os

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
    cwd = os.getcwd()
    usr = user.UserInput()
    usr.setInput()

    whole = "empty"
    for district in usr.inputDist:
        for year in usr.inputYr:
            d = dataSet.DataSet()
            d.getData(district, year)
            
            if len(usr.inputRoute) != 0:
                d.getRoute(usr.inputRoute)
            sect = d.data
            
            if whole == "empty":
                whole = sect
            else:
                whole = np.vstack((whole, sect))
    '''
    Column information
    first column is data year whole[:,0]
    distScore = whole[:,6]
    condScore = whole[:,7]
    rideScore = whole[:,8]
    ACP distress = whole[:,9:21]
    ACP IRI = whole[:,21:23], i.e. 21,22
    possibly not used, ACP raveling, flushing and severe and failure rutting, i.e. 15, 16, 19, 20
    TRM: whole[:, 2:6]
    '''
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
    print("\nDependent variable: %s" %colName[0])
    print("Covariates: %s\n" %colName[1:])

    # divide data into good & bad condition regions @ TRM-60
    start = list(whole[:,1]).index(55)
    end   = list(whole[:,1]).index(165)
    if usr.inputReg == "bad":
        print("Focusing on BAD region...")
        whole = whole[:start, :]
    elif usr.inputReg == "good":
        print("Focusing on GOOD region...")
        whole = whole[start:end, :]
    
    # remove 0's from CS
    if colName[0] == "CONDITION_SCORE":
        whole = removeCSZero(whole)

    # Single run experiment
    firstExp = framework.SingleRun(data=whole)
    firstExp.execute(updOpt='mcmcSml', iterMCMC=5000)

    # Cross validation experiment
    # whole = mainShuffle(whole, 1)
    # secondExp = framework.CrossValid(data=whole, foldPct=0.2, gap=usr.inputGap)
    # secondExp.execute(updOpt='mcmcSml', iterMCMC=5000)
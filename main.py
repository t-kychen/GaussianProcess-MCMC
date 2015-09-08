'''
Created on May 30, 2015

@author: Thomas
'''
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
    shuffle data to prevent from overfitting
    '''
    t = 0
        
    while t < tms:
        randIdx = random.sample(range(0, np.shape(data)[0]), np.shape(data)[0])
        data = data[randIdx,:]
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
    # acquire user input
    usr = user.UserInput()
    usr.setInput()
    #usr.getInput()

    # load data
    whole = "empty"
    for district in usr.inputDist:
        for year in usr.inputYr:
            d = dataSet.DataSet()
            d.getData(district, year)
            
            if len(usr.inputRoute) != 0:
                d.getRoute(usr.inputRoute)
            sect = d.data
            
            # stacking sect's in one big matrix - overall
            if whole == "empty":
                whole = sect
            else:
                whole = np.vstack((whole,sect))
    
    '''
    Column information
    #first column is data year whole[:,0]
    #distScore = whole[:,6]
    #condScore = whole[:,7]
    #rideScore = whole[:,8]
    #ACP distress = whole[:,9:21]
    #ACP IRI = whole[:,21:23], i.e. 21,22
    #possibly not used, ACP raveling, flushing and severe and failure rutting, i.e. 15, 16, 19, 20
    '''
    colExtracted = [7,0,6,8,9,10,12,13,14,17,18,21,22,2,3,4,5,24,25,26,27,28,29]
    
    whole = mainExtract(whole,colExtracted, "col")
    colName = np.reshape(d.feature, (1,len(d.feature)))
    colName = mainExtract(colName, colExtracted, "col")
    del d
    
    """
    # output data
    header = ["CONDITION_SCORE","DISTRESS_SCORE","RIDE_SCORE","ACP_PATCHING_PCT","ACP_FAILURE_QTY","ACP_BLOCK_CRACKING_PCT","ACP_ALLIGATOR_CRACKING_PCT","ACP_LONGITUDE_CRACKING_PCT","ACP_TRANSVERSE_CRACKING_QTY","ACP_RAVELING_CODE","ACP_FLUSHING_CODE","ACP_RUT_AUTO_SHALLOW_AVG_PCT","ACP_RUT_AUTO_DEEP_AVG_PCT","ACP_RUT_AUTO_SEVERE_AVG_PCT","ACP_RUT_AUTO_FAILURE_AVG_PCT","IRI_LEFT_SCORE","IRI_RIGHT_SCORE"]
    outputData(whole,header)
    F, p = pValue(whole)
    print("F: ", F)
    print("p value: ", p)
    """
    
    # adding TRM for each section
    trm =  whole[:,13]+whole[:,14]
    trm = np.reshape(trm, (len(trm),1))
    whole = np.hstack((whole,trm))
    colName = np.append(colName,["TRM"])
    
    # condition score is in the first column
    # year, ds, rs, acp distress(from idx 4 to 10), IRI(11,12), TRM(13 to 16), county, maint_sec, hwy_sys, pav_type, 18kip, aadt, new TRM
    colNotUsed = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
    whole = np.delete(whole, colNotUsed, 1)
    colName = np.delete(colName, colNotUsed, 0)
    print("\nDependent variable: %s" %colName[0])
    print("Covariates: %s\n" %colName[1:])
    
    """
    if percntHold != 0:
        fileName = str(datetime.now())[0:16]
        test_output = csv.writer(open("./results/" + fileName + "_test.csv","wb"))
        test_output.writerow("testing llk")
        test_output.writerow(["Plot Gap, Initial LL","Fold 1","Fold 2","Fold 3","Fold 4","Fold 5","Fold 6","Fold 7","Fold 8","Fold 9","Fold 10"])
        train_output = csv.writer(open("./results/" + fileName + "_train.csv","wb"))
        train_output.writerow("training llk")
        train_output.writerow(["Plot Gap, Initial LL","Fold 1","Fold 2","Fold 3","Fold 4","Fold 5","Fold 6","Fold 7","Fold 8","Fold 9","Fold 10"])
    
    overall_LK = []
    """
    
    # divide data into good & bad condition regions @ TRM-60
    threshold = list(whole[:,1]).index(60)
    
    if usr.inputReg == "bad":
        #bad region
        print("Focusing on BAD region...")
        whole = whole[:threshold,:]
    elif usr.inputReg == "good":
        #good region
        print("Focusing on GOOD region...")
        whole = whole[threshold:,:]
    else:
        print("Focusing on WHOLE region...")
    
    # remove 0 from CS
    if colName[0] == "CONDITION_SCORE":
        whole = removeCSZero(whole)

    # shuffle data
    #whole = mainShuffle(whole,2)
    print np.shape(whole)
    for gap in usr.inputGap:
        # Single run experiment
        for gap in [0.5,1,2,3]:
            firstExp = framework.SingleRun(data=whole, gap=gap)
            firstExp.execute(updOpt='mcmcAlt', iterMCMC=10000)
        
        '''
        # Cross validation experiment
        for gap in [0.5, 1, 2, 3, 4, 5]:
            secondExp = framework.CrossValid(data=whole, foldPct=0.1, gap=gap)
            secondExp.execute(updOpt='mcmcAlt', iterMCMC=100)
        '''
        
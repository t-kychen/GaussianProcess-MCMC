'''
Created on May 30, 2015

@author: Thomas
'''
import numpy as np

class DataSet(object):
    '''
    Data extraction class
    '''    
    
    def __init__(self, data=None):
        '''
        Constructor
        '''
        print("Loading data...")
        self.data = data
        self.data_string = None
        self.feature = None
    
    def getData(self, file_district, file_year):
        '''
        main function of the class
        '''
        file_name = "./" + file_district + "/" + file_district + "_" + file_year + ".dat"
        
        self.data = np.genfromtxt(file_name, delimiter=",", dtype="f8", usecols=range(1, 31), skip_header=1)
        self.data_string = np.genfromtxt(file_name, delimiter=",", dtype=None, usecols=range(0, 31))
        print("The following data %s %s has been successfully loaded!" %(file_district, file_year))
        
        # remove columns' with NA's
        deleted_col = self.removeColNA()
        self.feature = np.delete(self.data_string[0, 1:], deleted_col)

    def removeColNA(self):
        '''
        remove columns that are full of NA's
        '''
        num_row = np.shape(self.data)[0]
        num_col = np.shape(self.data)[1]
        col = 0
        record = 0
        deleted_list = []
        
        while col < num_col:
            if np.isnan(self.data[:,col]).tolist() == [True]*num_row:
                self.data = np.delete(self.data, col, 1)
                deleted_list.append(record)
            else:
                col += 1
            record += 1

        return deleted_list

    def getRoute(self, target_route):
        '''
        get row indices of the target route
        '''
        print("Highway selected: %s" %(target_route))
        
        route_col = self.data_string[1:,0]
        route_idx = []
        
        for i in range(len(route_col)):
            if route_col[i][0:len(target_route)] == target_route:
                route_idx.append(i)
        
        if len(route_idx) == 0:
            raise Exception("Selected route not found!")
        else:
            self.extractData(route_idx, "row")
            
    def extractData(self, col_idx, opt):
        '''
        get data on target route
        '''
        new_data = []
        
        if opt == "row":    #Extract data from target rows
            self.data = self.data.T
        
        for idx in col_idx:
            new_data.append(self.data[:,idx])
        new_data = np.asarray(new_data)
        
        if np.shape(new_data)[0] <= np.shape(new_data)[1]:
            new_data = new_data.T
        
        self.data = new_data
            
if __name__ == "__main__":
    print("DataSet class is implemented here!")

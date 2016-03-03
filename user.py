'''
Created on Jun 3, 2015

@author: Thomas
'''

class UserInput(object):
    '''
    User input class
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.inputDist = []
        self.inputYr = []
        self.inputReg = ""
        self.inputMdl = ""
        self.inputPcnt = 0
        self.inputRoute = ""
        self.inputGap = []
        self.inputPlot = ""
    
    def setInput(self):
        '''
        Set up user input by program, DEFAULT
        '''
        print("Setting default options...\n")
        
        self.inputDist = ["houston", "bryan"]
        self.inputYr = ["2008"]
        self.inputReg = "whole"   # whole, bad or good
        self.inputMdl = "GP"
        self.inputPcnt = 0
        self.inputRoute = "IH0045 L"
        self.inputGap = [1, 2]
        self.inputPlot = "y"
        
    def getInput(self):
        '''
        Get user option for modeling
        '''
        print("====Enter modeling options in the following prompts====")
        
        self.getDist()
        self.getYear()
        self.inputReg = raw_input("Region of IH 45 to be focused, whole, bad or good: ")
        self.inputMdl = raw_input("Models currently supported: OLS, LASSO, GP: ")
        self.inputPcnt = input(">>> Percentage of data held for cross validation, e.g. 0.1: ")
        self.inputRoute = raw_input(">>> Highway name, e.g. IH0045 L: ")
        self.getGap()
        self.inputPlot = raw_input(">>> Plot the result using matplotlib: y or n? ")
        
        print("===================End of user input===================\n")
        
    def getDist(self):
        '''
        User input district: houston, bryan
        '''
        district = raw_input(">>> District of data: houston, bryan or both? ")
        if district not in ["both","houston","bryan"]:
            raise Exception("District must be houston, bryan or both.")
        
        else:
            if district == "both":
                self.inputDist = ["houston","bryan"]
            else:
                self.inputDist.append(district)
    
    def getYear(self):
        '''
        User input year: 2008, 2009, 2010, 2011, 2012, 2013
        '''
        while True:
            year = raw_input(">>> Year of data, between 2008 and 2013: ")
            if year == "":
                break
            
            elif year not in ["","all","2008", "2009", "2010", "2011", "2012", "2013"]:
                raise Exception("Year must be between 2008 and 2013.")
        
            else:
                if year == "all":
                    self.inputYr = ["2008", "2009", "2010", "2011", "2012", "2013"]
                    break
                
                else:
                    self.inputYr.append(year)
        
    def getGap(self):
        '''
        User input gap measurement: 1, 2, 3 ...etc.
        '''
        while True:
            gap = raw_input(">>> Gap measurement, miles: ")
            if gap == "":
                break
            
            self.inputGap.append(float(gap))
    
if __name__ == "__main__":
    print("UserInput class is implemented here!")

'''
Created on Jun 3, 2015

@author: K.Y. Chen
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
        self.inputRoute = ""
        self.inputGap = []

    def setInput(self, district='both', year='2008', region='whole', route='IH0045 L', gapMin=1, gapMax=5):
        '''
        Set up user input by program, DEFAULT
        '''
        print("Setting default options...\n")

        if district == 'both':
            self.inputDist = ["houston", "bryan"]
        else:
            self.inputDist = district

        if year == '2008':
            self.inputYr += [year]
        else:
            self.getYear()

        self.inputReg = region   # whole, bad or good
        self.inputRoute = route

        self.inputGap = range(gapMin, gapMax+1)

    def getInput(self):
        '''
        Get user option for modeling
        '''
        print "====Enter modeling options in the following prompts===="
        
        self.getDist()
        self.getYear()
        self.inputReg = raw_input("Region of IH 45 to be focused, whole, bad or good: ")
        self.inputRoute = raw_input(">>> Highway name, e.g. IH0045 L: ")
        self.getGap()

        print "===================End of user input===================\n"
        
    def getDist(self):
        '''
        User input district: houston, bryan
        '''
        district = raw_input(">>> District of data: houston, bryan or both? ")
        if district not in ["both","houston","bryan"]:
            raise Exception("District must be houston, bryan or both.")
        
        else:
            if district == "both":
                self.inputDist = ["houston", "bryan"]
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

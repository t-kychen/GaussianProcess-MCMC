'''
Created on Jun 4, 2015

@author: Thomas
'''
import numpy as np
import kcGP
import MCMC
import triangle
import csv
import matplotlib.pyplot as plt
from kcGP import likK, covK, tools


class Framework(object):
    '''
    Framework class
    '''

    def __init__(self):
        '''
        Constructor
        '''
        # followings are input
        self.x = None
        self.y = None
        self.scaleOpt = None
        self.gap = None

        
        # followings are model
        self.mean = None
        self.cov  = None
        self.noise = None               # default noise: np.log(0.1)
        
        # followings are result
        self.testLLK = []
        self.trainLLK = []
    
    
    def setNoise(self, N):
        self.noise = N
    
    
    def execute(self):
        '''
        Main execution of the framework
        '''
        pass
    
    
    def setScaleOpt(self, scale):
        self.scaleOpt = scale
        print("Current data scaling option: %s" %(self.scaleOpt))
        
        
    def getScaleData(self):
        '''
        Scale data by centering, standardizing
        '''
        avg = np.mean(self.y, 0)
        stdev = np.std(self.y, 0)
        
        if self.scaleOpt == "ctr":
            print("Mean centering...")
            self.y = self.y - avg
            
        elif self.scaleOpt == "std":
            print("Standardizing...")
            self.y = (self.y - avg) / stdev
    
    
    def setGap(self, gap):
        '''
        Set the gap of training data
        '''
        self.gap = gap
        print("Current data gap: %f" %(self.gap))
        
        
    def getGapExtract(self):
        '''
        Extract data with certain gap
        
        @change the value of self.x and self.y
        '''
        if self.gap == 0.5:
            return None
        idxX = range(int(np.min(self.x)), int(np.max(self.x)), self.gap)
        idxY = []
        newX = []
        
        for i in range(len(idxX)):
            if idxX[i] not in self.x:
                continue
            else:
                newX.append(idxX[i])
                idxY.append(np.reshape(self.x, (len(self.x), )).tolist().index(idxX[i]))
        
        self.x = np.reshape(np.asarray(newX), (len(newX),1))
        self.y = self.y[idxY,]
    
    
    def runAlterMCMC(self, model, iters):
        '''
        Alternatively update latent variables and hyperparameters by ESS and HMC
        
        :param model: GP model instance
        :param iters: iterations of update
        
        :return: proposed f, propose hyps, log-likelihood
        '''
        # initial settings
        input_ff = np.zeros_like(self.x)
        curHyp = np.asarray([0.41,105,2])                              #curHyp = [lengthScale, signal, noise]
        var = np.array([input_ff, self.x, self.y, curHyp, 'ell'])

        # recording the posterior
        propLatent = np.zeros((np.shape(self.x)[0],iters))
        propHyp    = np.zeros((np.shape(curHyp)[0],iters))
        logLike = []
        
        # main loop of MCMC
        for i in range(iters):
            print '\nIteration: ', i+1
            
            # update latent variables
            curll, curff = MCMC.elliptical_slice(var)                   # explore p(f | D, theta) = 1/Z_f * L(f) * p(f)
            
            # update hyperparameters                                    # explore p(theta | f) = 1/Z_theta * N(f; 0, Sigma(theta)) * p(theta)
            print 'Length scale'
            curHyp[0], logp = MCMC.hmcK(x=curHyp[0], E=MCMC.logp_hyper, var=var, leapfrog=1, epsilon=0.002, nsamples=1)
            var[3] = curHyp

            print '\nSignal y'
            var[4] = 'sf2'
            curHyp[1], logp = MCMC.hmcK(x=curHyp[1], E=MCMC.logp_hyper, var=var, leapfrog=1, epsilon=0.2, nsamples=1)
            var[3] = curHyp
             
            print '\nNoise'
            var[4] = 'noise'
            curHyp[2], logp = MCMC.hmcK(x=curHyp[2], E=MCMC.logp_hyper, var=var, leapfrog=1, epsilon=0.01, nsamples=1)
            var[3] = curHyp
            
            # update var for HMC
            var = np.array([curff, self.x, self.y, curHyp, 'ell'])

            print '================='

            logLike.append(curll)
            propLatent[:,i] = curff.reshape((np.shape(curff)[0],))
            propHyp[:,i]    = curHyp.reshape((np.shape(curHyp)[0],))
            
        
        return propLatent, propHyp, logLike

    
    def runSimulMCMC(self):
        '''
        Simultaneously update latent variables and hyperparameters by Surrogate slice sampling
        '''
        raise NotImplementedError


class SingleRun(Framework):
    '''
    First framework: Gaussian Process without cross validation
    '''
    def __init__(self, data, gap):
        # followings are input
        if data is None:
            raise Exception('No data is given.')
        self.x = data[:,1:]
        self.y = np.reshape(data[:,0], (np.shape(data)[0],1))
        
        self.setScaleOpt(None)
        self.setGap(gap)
        
        # followings are model
        self.mean = kcGP.meanK.Zero()
        self.cov  = kcGP.covK.RBF()
        self.noise = None               # default noise: 0.1
        
        # followings are result
        self.testLLK = []
        self.trainLLK = []

        self.name = 'First framework: no cross validation'
        
        
    def execute(self, updOpt=None, iterMCMC=1000):
        self.getScaleData()
        self.getGapExtract()
        xs = np.arange(np.min(self.x),np.max(self.x),0.1)  # create test x for every 0.1 miles
        xs = np.reshape(xs,(xs.shape[0],1))

        
        model = kcGP.gpK.GPR()
        model.setPrior(self.mean, self.cov)
        if not self.noise is None:
            model.setNoise(self.noise)
        model.getPosterior(self.x, self.y)
#         model.predict(xs)
#         model.plot()
        
        assert updOpt is not None, 'Please choose either optimization or MCMC'
        if updOpt == 'opt':                 # optimize log-likelihood w.r.t. hyperparameters
            model.optimize(self.x, self.y)
        elif updOpt == 'mcmcAlt':           # alternatively, run ESS and HMC
            propF, propHyp, llk = self.runAlterMCMC(model, iterMCMC)
        elif updOpt == 'mcmcSml':           # simultaneously, run Surrogate slice sampling
            propF, propHyp, llk = self.runSimulMCMC(model, iterMCMC)
        
        
        self.cov  = kcGP.covK.RBF(log_ell=np.log(propHyp[0,-1]), log_sigma=np.log(propHyp[1,-1]))
        model.setNoise(np.log(propHyp[2,-1]))
        model.setPrior(self.mean, self.cov)
        model.getPosterior(self.x, self.y)
        model.predict(xs)
        model.plot()
        print "Length scale for prediction:", np.exp(model.covfunc.hyp[0])
        print "Signal y for prediction:", np.exp(model.covfunc.hyp[1])
        print 'Noise: ', np.exp(model.likfunc.hyp[0])
                        
        # output result for plot
        with open('Proposed_hyper_llk.csv', 'wb') as h:
            writer = csv.writer(h)
            writer.writerow(range(iterMCMC))
            writer.writerows(propHyp)
            writer.writerow(llk)
 
        with open('Proposed_F.csv', 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(range(iterMCMC))
            writer.writerows(propF)


class CrossValid(Framework):
    '''
    Second framework: Gaussian Process with cross validation
    '''
    def __init__(self, data, foldPct, gap=0.5):
        # followings are input
        if data is None:
            raise Exception('No data is given.')
        self.x = data[:,1:]
        self.y = np.reshape(data[:,0], (np.shape(data)[0],1))
        
        self.setScaleOpt(None)
        self.setGap(gap)
        
        # followings are model
        self.mean = kcGP.meanK.Zero()
        self.cov  = kcGP.covK.RBF()
        self.noise = None               # default noise: np.log(0.1)
        
        # followings are result
        self.testLLK = []
        self.trainLLK = []

        self.name = 'Second framework: ' + str(int(1/foldPct)) + ' folds cross validation'
        self.foldPct = foldPct
        
        
    def execute(self, updOpt=None, iterMCMC=1000):
        self.getScaleData()
        
        try:
            numFold = int(1/self.foldPct)
        except ZeroDivisionError, e:
            print("Zero fold is not valid. Run SingleRun instead")
            numFold = 1
            self.foldPct = 1
        else:
            print("Running %d fold cross validation" %numFold)
            llk = []
            originalX = self.x[:]
            originalY = self.y[:]
            
            # main loop of cross validation
            for fold in range(0, numFold):
                print("%d fold" %(fold+1))
                model = kcGP.gpK.GPR()
                
                self.x, self.y = originalX, originalY
                trainX, trainY, testX, testY = self.getFoldData(fold)
                self.x, self.y = trainX, trainY
                self.getGapExtract()                # extract gapped training data
            
                model.setPrior(self.mean, self.cov)
                if not self.noise is None:
                    model.setNoise(self.noise)
                model.getPosterior(self.x, self.y)
            
                assert updOpt is not None, 'Please choose either optimization or MCMC'
                if updOpt == 'opt':                 # optimize log-likelihood w.r.t. hyperparameters
                    model.optimize(self.x, self.y)
                elif updOpt == 'mcmcAlt':           # alternatively, run ESS and HMC
                    propF, propHyp, junk = self.runAlterMCMC(model, iterMCMC)
                elif updOpt == 'mcmcSml':           # simultaneously, run Surrogate slice sampling
                    propF, propHyp, junk = self.runSimulMCMC(model, iterMCMC)
                
                # make prediction
                xs = np.reshape(testX,(np.shape(testX)[0],1))
                lp, ymu, ys2 = MCMC.predictMCMC(self.x, self.y, xs, propF[:,-1], 
                                                model.posterior.L, model.meanfunc, model.covfunc, model.likfunc, ys=testY)
                print('Log-likelihood: %d' %(np.sum(lp)))
                llk.append(np.sum(lp))

            #MCMC.plotMCMC(range(1,numFold+1), llk, iter=True)
            print llk, '\n'
   
    
    def getFoldData(self, fold):
        '''
        Separate data into training and test according to the fold
        
        @param fold: the fold of the data
        @return: train x, y and test x, y
        '''
        oneFold = int(np.shape(self.x)[0]*self.foldPct)
        
        testID  = range(fold*oneFold, (fold+1)*oneFold)
        trainID = range(0, np.shape(self.x)[0])
        del trainID[fold*oneFold:(fold+1)*oneFold]
        
        # order: train X, train Y, test X, test Y
        return self.x[trainID,:], self.y[trainID,0], self.x[testID,:], self.y[testID,0]
    
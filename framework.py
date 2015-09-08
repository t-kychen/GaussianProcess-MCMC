'''
Created on Jun 4, 2015

@author: Thomas
'''
import numpy as np
import kcGP
import MCMC
from kcGP import likK, covK, tools
from pyhmc import hmc
import triangle
import csv

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
        
        @param model: GP model instance
        @param iters: iterations of update
        
        @return: proposed f, propose hyps, log-likelihood
        '''
        # initial settings
        input_ff = np.zeros((np.shape(self.x)[0],1))
        curHyp = np.asarray([1.,1.,1])                              #curHyp = [lengthScale, signal, noise]
        var = np.array([input_ff, self.x, self.y, curHyp, 'ell'])

        # recording the results
        propLatent = np.zeros((np.shape(self.x)[0],iters))
        propHyp    = np.zeros((np.shape(curHyp)[0],iters))
        logLike = []
        
        # main loop of MCMC
        for i in range(iters):
            # update hyperparameters
            print '\nIteration: ', i+1
            '''
            print 'Length scale'
            curHyp[0], logp = MCMC.hmcK(q=curHyp[0], E=MCMC.logp_hyper, var=var, leapfrog=1, epsilon=0.2, nsamples=1)
            var[3] = curHyp
            
            print 'Signal y'
            var[4] = 'sf2'
            curHyp[1], logp = MCMC.hmcK(q=curHyp[1], E=MCMC.logp_hyper, ivar=ivar, leapfrog=1, epsilon=0.02, nsamples=1)
            
            print 'Noise'
            var[4] = 'noise'
            curHyp[2], logp = MCMC.hmcK(q=curHyp[2], E=MCMC.logp_hyper, var=var, leapfrog=1, epsilon=0.1, nsamples=1)
            var[3] = curHyp
            
            # change covariance
            model.covfunc.hyp[0] = np.log(curHyp[0])
            model.setNoise(np.log(curHyp[2]))
            model.getPosterior(self.x, self.y)
            '''
            # update latent variables
            curll, curff = MCMC.elliptical_slice(var)
            input_ff = curff
            print 'Proposed ff', curff[0,0]
            
            # update ivar for HMC
            var = np.array([input_ff, self.x, self.y, curHyp, 'ell'])

            propLatent[:,i] = curff.reshape((np.shape(curff)[0],))
            propHyp[:,i]    = curHyp.reshape((np.shape(curHyp)[0],))
            logLike.append(curll)
            print 'Model covariance hyp: ', np.exp(model.covfunc.hyp[0]), np.exp(model.covfunc.hyp[1]*2)
            print 'Model llk noise: ', np.exp(2.*model.likfunc.hyp[0])

        
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
        self.noise = None               # default noise: np.log(0.1)
        
        # followings are result
        self.testLLK = []
        self.trainLLK = []

        self.name = 'First framework: no cross validation'
        
        
    def execute(self, updOpt=None, iterMCMC=1000):
        self.getScaleData()
        self.getGapExtract()
        
        model = kcGP.gpK.GPR()
        model.setPrior(self.mean, self.cov)
        if not self.noise is None:
            model.setNoise(self.noise)
        model.getPosterior(self.x, self.y)
        print 'Length scale (initial): ', np.exp(model.covfunc.hyp[0])
        
        assert updOpt is not None, 'Please choose either optimization or MCMC'
        if updOpt == 'opt':                 # optimize log-likelihood w.r.t. hyperparameters
            model.optimize(self.x, self.y)
        elif updOpt == 'mcmcAlt':           # alternatively, run ESS and HMC
            print 'MCMC iterations: ', iterMCMC
            propF, propHyp, llk = self.runAlterMCMC(model, iterMCMC)
        elif updOpt == 'mcmcSml':           # simultaneously, run Surrogate slice sampling
            propF, propHyp, llk = self.runSimulMCMC(model, iterMCMC)
        
        # output result for plot
        with open('Proposed_hyper.csv', 'wb') as f:
            writer = csv.writer(f)
            writer.writerows(propHyp)
            writer.writerow(propF)
            writer.writerow(llk)

        print
        print 'Proposed length scale: ', np.exp(propHyp[0,-1])
        print 'Proposed noise: ', np.exp(2.*propHyp[2,-1])
        print 'Associated log-likelihood: ', llk[-1]
                
        # plot sampling log-likelihood
        MCMC.plotMCMC(range(iterMCMC), llk, iter=True)        
        
        # make sure using the right hyperparameters
        print 'Length scale for prediction: ', np.exp(model.covfunc.hyp[0])
        print 'Noise: ', np.exp(2. * model.likfunc.hyp[0])

        # make prediction        
        xs = np.arange(np.min(self.x),np.max(self.x),0.1)  # create test x for every 0.1 miles
        xs = np.reshape(xs,(xs.shape[0],1))
        self.cov  = kcGP.covK.RBF(log_ell=np.log(propHyp[0,-1]), log_sigma=np.log(propHyp[1,-1]/2.))
        #model.setNoise(propHyp[2,-1])
        model.setPrior(self.mean, self.cov)
        model.getPosterior(self.x, self.y)
        ymu, ys2, fmu, fs2, junk = model.predict(xs=xs)
        #junk, ymu, ys2 = MCMC.predictMCMC(self.x, self.y, xs, propF[:,-1], model.posterior.L, model.meanfunc, model.covfunc, model.likfunc)
        
        # plotting prediction result
        MCMC.plotMCMC(self.x, self.y, iter=False, input_xs=xs, input_ym=ymu, input_ys=ys2)
        
        print '\nComparison: length scale=1, signal=1'
        self.cov  = kcGP.covK.RBF(log_ell=0., log_sigma=0.)
        model.setPrior(self.mean, self.cov)
        #model.setNoise(np.log(0.1))
        model.getPosterior(self.x, self.y)
        ymu, ys2, fmu, fs2, junk = model.predict(xs=xs)
        MCMC.plotMCMC(self.x, self.y, iter=False, input_xs=xs, input_ym=ymu, input_ys=ys2)



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
    
'''
Created on Jun 4, 2015

@author: Thomas
'''
import os
import kcGP
import csv
import numpy as np
from MCMC import elliptical_slice, surrogate_slice_sampling, infMCMC


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

        # followings are result
        self.testLLK = []
        self.trainLLK = []
    
    
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

        
    def getGapExtract(self, gap):
        '''
        Extract data with certain gap
        
        @change the value of self.x and self.y
        '''
        print 'Current gap %f' %(gap)
        if gap == 0.5:
            return None
        idxX = range(int(np.min(self.x)), int(np.max(self.x)), gap)
        idxY = []
        newX = []
        
        for i in range(len(idxX)):
            if idxX[i] not in self.x:
                continue
            else:
                newX.append(idxX[i])
                idxY.append(np.reshape(self.x, (len(self.x), )).tolist().index(idxX[i]))
        
        self.x = np.reshape(np.asarray(newX), (len(newX), 1))
        self.y = self.y[idxY, ]
    
    
    def runAlterMCMC(self, iters):
        '''
        Alternatively update latent variables and hyperparameters by ESS and HMC
        
        :param model: GP model instance
        :param iters: iterations of update
        
        :return: proposed f, propose hyps, log-likelihood
        '''
        y = self.y.reshape((self.y.shape[0],))
        input_ff = np.zeros_like(y)
        curHyp = np.asarray([0.5, 15.])
        sn = 0.3
        var = np.array([input_ff, self.x, y, curHyp])

        histF = np.zeros((y.shape[0], iters))
        histHyp = np.zeros((curHyp.shape[0], iters))
        logLike = []

        # main loop of MCMC
        for i in range(iters):
            print 'Iteration: ', i+1
            
            # update latent variables
            llk, propF = elliptical_slice(var, sn)                   # explore p(f | D, theta) = 1/Z_f * L(f) * p(f)
            print 'log likelihood ', llk
            
            # update hyperparameters                                    # explore p(theta | f) = 1/Z_theta * N(f; 0, Sigma(theta)) * p(theta)
            #  curHyp, logp = MCMC.hmcK(x=curHyp, E=MCMC.logp_hyper, var=var, leapfrog=1, epsilon=np.asarray([0.004, 0.01, 0.005]), nsamples=1)
            
            # update var
            var = np.asarray([propF, self.x, y, curHyp])

            print '================='

            logLike.append(llk)
            histF[:, i] = propF
            histHyp[:, i] = curHyp.reshape((curHyp.shape[0], ))

        return histF, histHyp, llk, sn

    
    def runSimulMCMC(self, iters):
        '''
        Simultaneously update latent variables and hyperparameters by Surrogate slice sampling
        '''
        # initial settings
        y = self.y.reshape((self.y.shape[0],))
        hyp = np.asarray([0.5, 15.])                              # curHyp = [lengthScale, signal]
        sn  = 0.3
        f = np.zeros_like(y)
        var = np.array([f, self.x, y, hyp])

        # recording the posterior
        histF = np.zeros((y.shape[0], iters))
        histHyp = np.zeros((hyp.shape[0]+1, iters))
        
        # main loop of MCMC
        for i in range(iters):
            print '\nIteration: ', i+1
            propF, propHyp = surrogate_slice_sampling(var=var, sn=sn, scale=np.asarray([2.5, 2.5, 1.]), opt=0)
            var[0] = propF
            var[3] = propHyp[0:2]
            sn = propHyp[2]

            print 'hyp', propHyp
            print '================='
            
            histF[:, i] = propF
            histHyp[:, i] = propHyp
            
        return histF, histHyp, sn


    def output(self, model, iterMCMC, gap, histHyp=None, histF=None, llk=None):
        '''
        (only for MCMC) output proposed hyp-parameters, f's and llk's
        :param model: kcGP object, GP model used
        :param iterMCMC: int, number of MCMC iterations
        :param histHyp: ndarray, proposed hyper-parameters
        :param histF: ndarray, proposed latent f's
        :param llk: float, log likelihood of each proposed f's

        :return hyp, f: csv file
        '''
        cwd = os.getcwd()
        if not histHyp is None:
            with open(cwd+'/output/hyp_gap'+str(gap)+'.csv', 'wb') as h:
                writer = csv.writer(h)
                writer.writerow(["ll", "sf2", "sn"])
                writer.writerows(histHyp)

        if not histF is None:
            with open(cwd+'/output/f_gap'+str(gap)+'.csv', 'wb') as f:
                writer = csv.writer(f)
                writer.writerow(model.x)
                writer.writerow(model.y)
                xy = np.tile(np.hstack((model.x, model.y)), (iterMCMC, 1))
                writer.writerows(histF)

        if not llk is None:
            with open(cwd+'/output/llk.csv', 'wb') as k:
                writer = csv.writer(k)
                writer.writerow(['gap', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10'])
                writer.writerows(llk)



class SingleRun(Framework):
    '''
    First framework: singlue-run Gaussian process
    '''
    def __init__(self, data, gap):
        # followings are input
        if data is None:
            raise Exception('No data is given.')
        self.x = data[:, 1:]
        self.y = np.reshape(data[:, 0], (np.shape(data)[0], 1))
        
        self.setScaleOpt(None)
        self.setGap(gap)
        
        self.name = 'First framework: single-run GP'
        
        
    def execute(self, updOpt=None, iterMCMC=1000):
        self.getScaleData()
        self.getGapExtract(self.gap[0])
        xs = np.arange(np.min(self.x), np.max(self.x), 0.1)  # create test x for every 0.1 miles
        xs = np.reshape(xs, (xs.shape[0], 1))
        
        model = kcGP.gpK.GPR()
        assert updOpt is not None, 'Please choose either optimization or MCMC'
        if updOpt == 'opt':                     # optimize log-likelihood w.r.t. hyperparameters
            model.optimize(self.x, self.y)
            print np.exp(model.covfunc.hyp[0]), np.exp(model.covfunc.hyp[1])
            print np.exp(model.likfunc.hyp)

        elif updOpt[0:4] == 'mcmc':
            if updOpt == 'mcmcAlt':             # alternatively, run ESS and HMC
                histF, histHyp, llk, sn = self.runAlterMCMC(iterMCMC)
            
            elif updOpt == 'mcmcSml':           # simultaneously, run Surrogate slice sampling
                histF, histHyp, sn = self.runSimulMCMC(iterMCMC)

            # prediction
            ll = np.mean(histHyp[0, -0.3*iterMCMC:])
            sf2 = np.mean(histHyp[1, -0.3*iterMCMC:])
            print 'mean of posterior ll %.3f, sf %.3f' %(ll, sf2)
            print 'last posterior ll %.3f, sf %.3f' %(histHyp[0, -1], histHyp[1, -1])

            covMCMC = kcGP.covK.RBF(np.log(ll), np.log(sf2))
            model.setPrior(kernel=covMCMC)
            model.setNoise(np.log(sn))
            model.getPosterior(self.x, self.y)

            # output proposed f's and hyper-parameters
            self.output(model, iterMCMC, histHyp.T, histF.T)

            model.xs = xs
            ym, ys_lw, ys_up, fmu, fs2 = infMCMC(xs, histF[:, -500:], model)
            model.ym = ym + np.mean(self.y)
            model.plot(ys_lw+np.mean(self.y), ys_up+np.mean(self.y))


class CrossValid(Framework):
    '''
    Second framework: Gaussian Process with cross validation
    '''
    def __init__(self, data, foldPct, gap=0.5):
        # followings are input
        if data is None:
            raise Exception('No data is given.')
        self.x = data[:, 1:]
        self.y = np.reshape(data[:, 0], (np.shape(data)[0], 1))
        
        self.setScaleOpt(None)
        self.setGap(gap)
        
        # followings are result
        self.testLLK = None

        self.name = 'Second framework: GP with %d folds cross validation' %(1/foldPct)
        self.foldPct = foldPct
        
        
    def execute(self, updOpt=None, iterMCMC=1000):
        self.getScaleData()
        
        try:
            numFold = int(1/self.foldPct)
        except ZeroDivisionError, e:
            print "Zero fold is not valid. Run SingleRun instead"
            numFold = 1
            self.foldPct = 1
        else:
            print "Running %d fold cross validation" %(numFold)
            originalX = self.x[:]
            originalY = self.y[:]

            for gap in self.gap:
                gapLLK = []
                histF = None
                histHyp = None
                for fold in range(numFold):
                    foldLLK = []
                    model = kcGP.gpK.GPR()
                
                    self.x, self.y = originalX, originalY
                    self.x, self.y, testX, testY = self.getFoldData(fold)
                    self.getGapExtract(gap)
            
                    assert updOpt is not None, 'Please choose either optimization or MCMC'
                    if updOpt == 'opt':
                        model.optimize(self.x, self.y)

                    elif updOpt[0:4] == 'mcmc':
                        if updOpt == 'mcmcAlt':
                            foldF, foldHyp, llk, sn = self.runAlterMCMC(iterMCMC)

                        elif updOpt == 'mcmcSml':
                            foldF, foldHyp, sn = self.runSimulMCMC(iterMCMC)
                    if histF is None or histHyp is None:
                        histF = foldF.T
                        histHyp = foldHyp.T
                    else:
                        histF = np.vstack((histF, foldF.T))
                        histHyp = np.vstack((histHyp, foldHyp.T))

                    model.xs = testX
                    selected_sample = range(iterMCMC*3/4, iterMCMC, 100)
                    for i in selected_sample:
                        ll = foldHyp[0, i]
                        sf = foldHyp[1, i]
                        sn = foldHyp[2, i]
                        covMCMC = kcGP.covK.RBF(np.log(ll), np.log(sf))
                        model.setPrior(kernel=covMCMC)
                        model.setNoise(np.log(sn))
                        model.getPosterior(self.x, self.y)

                        ym, ys_lw, ys_up, fmu, fs2, lp = infMCMC(testX, foldF[:, i].reshape((foldF.shape[0], 1)), model)
                        foldLLK.append(lp)

                    gapLLK.append(np.mean(foldLLK))
                if self.testLLK is None:
                    self.testLLK = [gap]+gapLLK
                else:
                    self.testLLK = np.vstack((self.testLLK, [gap]+gapLLK))

                self.output(model, iterMCMC, gap, histHyp, histF)

            self.testLLK = np.reshape(self.testLLK, (len(self.gap), numFold+1))
            self.output(model, iterMCMC, gap, histHyp=None, histF=None, llk=self.testLLK)


    
    def getFoldData(self, fold):
        '''
        Separate data into training and test according to the fold
        :param fold: int, the fold(th) of the data

        :return: train x, y and test x, y
        '''
        oneFold = int(self.x.shape[0]*self.foldPct)
        
        testID  = range(fold*oneFold, (fold+1)*oneFold)
        trainID = range(0, self.x.shape[0])

        return self.x[trainID, :], self.y[trainID, 0], self.x[testID, :], self.y[testID, 0]
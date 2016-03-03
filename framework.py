'''
Created on Jun 4, 2015

@author: Thomas
'''
import os
import kcGP
import csv
import numpy as np
import pandas as pd
from kcMCMC import sdsK

class Framework(object):
    """Framework class

    Parameters
    ----------
    data: ndarray with shape (n_samples, n_features+1)
        target & training samples
    foldPct: float
        percentage of data to be held out for cross validation
    gap: float
        interval between observations
    """
    def __init__(self, data, foldPct=None, gap=None):
        if data is None:
            raise Exception('No data is given.')
        self.x = data[:, 1:]
        self.y = np.reshape(data[:, 0], (np.shape(data)[0], 1))

        self.setScaleOpt(None)

        self.testLLK = None

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

    def runAlterMCMC(self, iters):
        """Alternatively update latent variables and hyperparameters by ESS and HMC

        :param model: GP model instance
        :param iters: iterations of update
        """
        y = self.y.reshape((self.y.shape[0],))
        input_ff = np.zeros_like(y)
        curHyp = np.asarray([0.5, 15.])
        sn = 0.1
        var = np.array([input_ff, self.x, y, curHyp])

        histF = np.zeros((y.shape[0], iters))
        histHyp = np.zeros((curHyp.shape[0], iters))
        logLike = []

        # main loop of MCMC
        for i in range(iters):
            print 'Iteration: ', i+1

            # update latent variables
            llk, propF = sdsK.elliptical_slice(var, sn)                 # explore p(f | D, theta) = 1/Z_f * L(f) * p(f)
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
        """Simultaneously update latent variables and hyper-parameters by surrogate slice sampling
        """
        y = self.y.reshape((self.y.shape[0],))
        hyp = np.asarray([1., 10., 1.])                              # curHyp = [lengthScale, signal, noise]
        f = np.zeros_like(y)

        histF = np.zeros((y.shape[0], iters))
        histHyp = np.zeros((hyp.shape[0], iters))
        for i in range(iters):
            print 'Iteration: ', i+1
            propF, propHyp = sdsK.surrogate_slice_sampling(f, self.x, y, hyp, scale=np.asarray([5., 5., 3.]))
            propF = sdsK.elliptical_slice(propF, self.x, y, propHyp)
            print 'll=%.3f, sf=%.3f, sn=%.3f' %(propHyp[0], propHyp[1], propHyp[2])
            f = propF
            hyp = propHyp


            histF[:, i] = propF
            histHyp[:, i] = propHyp
            
        return histF, histHyp

    def output(self, model, gap=0.5, histHyp=None, histF=None, llk=None):
        """(only for MCMC) Output proposed hyp-parameters, f's and llk's

        Parameters
        ----------
        model: kcGP instance
            Gaussian process model in use
        gap: float
            interval between observations
        histHyp: ndarray with shape (n_samples, n_mcmc_iters)
            proposed hyper-parameters
        histF: ndarray with shape (n_samples, n_mcmc_iters)
            proposed latent f's
        llk: ndarray with shape (n_gaps, n_folds)
            log likelihood of each proposed f's
        """
        if not (histHyp is None):
            with open('./output/hyp_gap'+str(gap)+'.csv', 'wb') as h:
                writer = csv.writer(h)
                writer.writerow(["ll", "sf2", "sn"])
                writer.writerows(histHyp)

        if not (histF is None):
            with open('./output/f_gap'+str(gap)+'.csv', 'wb') as f:
                writer = csv.writer(f)
                first_row = range(1, histF.shape[1]+1)
                first_row.append("x")
                first_row.append("y")
                writer.writerow(first_row)
                x = self.x.reshape((self.x.shape[0], 1))
                y = self.y.reshape((self.y.shape[0], 1))
                xy = np.hstack((x, y))
                writer.writerows(np.hstack((histF, xy)))

        if not (llk is None):
            with open('./output/llk.csv', 'wb') as k:
                writer = csv.writer(k)
                writer.writerow(['gap', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10'])
                writer.writerows(llk)

class SingleRun(Framework):
    """First framework: singlue-run Gaussian process
    """
    def __init__(self, data):
        super(SingleRun, self).__init__(data, foldPct=None, gap=None)
        self.name = 'First framework: single-run GP'

    def execute(self, updOpt=None, iterMCMC=1000):
        self.getScaleData()
        xs = np.arange(np.min(self.x), np.max(self.x), 0.1)
        xs = np.reshape(xs, (xs.shape[0], 1))
        model = kcGP.gpK.GPR()
        assert updOpt is not None, 'Please choose either optimization or MCMC'
        if updOpt == 'opt':
            model.optimize(self.x, self.y)

        elif updOpt[0:4] == 'mcmc':
            if updOpt == 'mcmcAlt':
                histF, histHyp, llk, sn = self.runAlterMCMC(iterMCMC)

            elif updOpt == 'mcmcSml':
                histF, histHyp = self.runSimulMCMC(iterMCMC)

            idx_100 = range(iterMCMC*3/4-1, iterMCMC, 100)
            ll = np.mean(histHyp[0, idx_100])
            sf = np.mean(histHyp[1, idx_100])
            sn = np.mean(histHyp[2, idx_100])
            print 'mean posterior of last 500 with gap 100: ll %.3f, sf %.3f, sn %.3f' %(ll, sf, sn)
            print 'last posterior: ll %.3f, sf %.3f, sn %.3f' %(histHyp[0, -1], histHyp[1, -1], histHyp[2, -1])

            covMCMC = kcGP.covK.RBF(np.log(ll), np.log(sf))
            model.setPrior(kernel=covMCMC)
            model.setNoise(np.log(sn))
            model.getPosterior(self.x, self.y)

            self.output(model, histHyp=histHyp.T, histF=histF)

class CrossValid(Framework):
    """Second framework: Gaussian process with cross validation

    Parameters
    ----------
    data: ndarray with shape (n_samples, n_features+1)
        target & training samples
    foldPct: float
        percentage of data to be held out for cross validation
    gap: float
        interval of observations
    """
    def __init__(self, data, foldPct, gap):
        super(CrossValid, self).__init__(data, foldPct, gap)
        self.name = 'Second framework: GP with %d folds cross validation' %(1/foldPct)
        self.gap = gap
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
                    print 'Gap: %r, fold %d' %(gap, fold+1)
                    foldLLK = []
                    model = kcGP.gpK.GPR()
                
                    self.x, self.y = originalX, originalY
                    self.x, self.y, valX, valY = self.getFoldData(fold, gap, 5)

                    assert updOpt is not None, 'Please choose either optimization or MCMC'
                    if updOpt == 'opt':
                        model.optimize(self.x, self.y)

                    elif updOpt[0:4] == 'mcmc':
                        foldF, foldHyp = self.runSimulMCMC(iterMCMC)

                    if histF is None or histHyp is None:
                        histF = foldF
                        histHyp = foldHyp
                    else:
                        histF = np.hstack((histF, foldF))
                        histHyp = np.vstack((histHyp, foldHyp))

                #     model.xs = valX
                #     selected_sample = range(iterMCMC*3/4, iterMCMC, 100)
                #     for i in selected_sample:
                #         ll = foldHyp[0, i]
                #         sf = foldHyp[1, i]
                #         sn = foldHyp[2, i]
                #         covMCMC = kcGP.covK.RBF(np.log(ll), np.log(sf))
                #         model.setPrior(kernel=covMCMC)
                #         model.setNoise(np.log(sn))
                #         model.getPosterior(self.x, self.y)
                #
                #         ym, ys_lw, ys_up, fmu, fs2, lp = sdsK.inf_mcmc(valX, foldF[:, i].reshape((foldF.shape[0], 1)), model, valY)
                #         foldLLK.append(lp)
                #
                #     gapLLK.append(np.mean(foldLLK))
                # if self.testLLK is None:
                #     self.testLLK = [gap]+gapLLK
                # else:
                #     self.testLLK = np.vstack((self.testLLK, [gap]+gapLLK))

                self.output(model, gap, histHyp, histF)

            # self.testLLK = np.reshape(self.testLLK, (len(self.gap), numFold+1))
            # self.output(model, gap, histHyp=None, histF=None, llk=self.testLLK)


    def getFoldData(self, fold, gap, window):
        """Separate data into training and test according to the fold

        Parameters
        ----------
        fold: int
            the fold+1(th) of the data
        gap: int
            the number of data points masked for testing
        window: int
            the length of data points reserved for training

        Return
        __________
        train_x, train_y, test_x, test_y
        """
        test_id = []
        for i in range(self.x.shape[0]/(gap+window)):

            test_id = np.append(test_id, fold+np.arange(gap)+(gap+window)*i)

        train_id = np.delete(np.array(range(self.x.shape[0])), test_id)
        test_id = test_id.astype('int')

        return self.x[train_id, :], self.y[train_id, 0], self.x[test_id, :], self.y[test_id, 0]
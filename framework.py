'''
Created on Jun 4, 2015

@author: K.Y. Chen
'''
import kcGP
import csv
import numpy as np
import statsmodels.api as sm
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
    def __init__(self, data, window=None, gap=None):
        if data is None:
            raise Exception('No data is given.')
        self.x = data[:, 1:]
        self.y = np.reshape(data[:, 0], (np.shape(data)[0], 1))

        self.setScaleOpt(None)

        # self.testLLK = {}

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

    def runSimulMCMC(self, iters):
        """Simultaneously update latent variables and hyper-parameters by surrogate slice sampling
        """
        y = self.y.reshape((self.y.shape[0],))
        propHyp = np.asarray([1., 10., 1.2])                              # curHyp = [lengthScale, signal, noise]
        propF = np.zeros_like(y)

        histF = np.zeros((y.shape[0], iters))
        histHyp = np.zeros((propHyp.shape[0], iters))
        for i in range(iters):
            propF, propHyp = sdsK.surrogate_slice_sampling(propF, self.x, y, propHyp, scale=np.asarray([10., 10., 5.]),
                                                           iter=i)
            # propF = sdsK.elliptical_slice(propF, self.x, y, propHyp)
            print 'Iteration: %r: ll=%.3f, sf=%.3f, sn=%.3f' %(i+1, propHyp[0], propHyp[1], propHyp[2])

            histF[:, i] = propF
            histHyp[:, i] = propHyp
            
        return histF, histHyp

    def output(self, gap=0, histHyp=None, histF=None, llk=None):
        """(only for MCMC) Output proposed hyp-parameters, f's and llk's

        Parameters
        ----------
        gap: integer
            interval between observations
        histHyp: ndarray with shape (n_samples, n_mcmc_iters)
            proposed hyper-parameters
        histF: ndarray with shape (n_samples, n_mcmc_iters)
            proposed latent f's
        llk: array with length = # CV fold
            log likelihood of each proposed f's
        """
        if not (histHyp is None):
            with open('./output/hypGap'+str(gap)+'.csv', 'w') as h:
                writer = csv.writer(h)
                writer.writerow(["ll", "sf2", "sn"])
                writer.writerows(histHyp)

        if not (histF is None):
            with open('./output/fGap'+str(gap)+'.csv', 'w') as f:
                first_row = range(1, histF.shape[1]+1)
                first_row.append("x")
                first_row.append("y")

                writer = csv.writer(f)
                writer.writerow(first_row)
                x = self.x.reshape((self.x.shape[0], 1))
                y = self.y.reshape((self.y.shape[0], 1))
                xy = np.hstack((x, y))
                writer.writerows(np.hstack((histF, xy)))

        if not (llk is None):
            with open('./output/llkGap'+str(gap)+'.csv', 'w') as k:
                header = ['gap']
                for i in range(len(llk)):
                    header.append(str(i))

                writer = csv.writer(k)
                writer.writerow(header)
                writer.writerow([gap]+llk)

        return 0

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
        test_id = test_id[test_id<self.x.shape[0]].astype('int')

        return self.x[train_id, :], self.y[train_id, 0], self.x[test_id, :], self.y[test_id, 0], test_id

class singleRun(Framework):
    """First framework: singlue-run Gaussian process
    """
    def __init__(self, data):
        super(singleRun, self).__init__(data, window=None, gap=None)
        self.name = 'First framework: single-run GP'

    def execute(self, updOpt=None, iterMCMC=1000):
        self.getScaleData()
        model = kcGP.gpK.GPR()
        assert updOpt is not None, 'Please choose either optimization or MCMC'
        if updOpt == 'opt':
            model.optimize(self.x, self.y)

        elif updOpt == 'mcmcSml':
            histF, histHyp = self.runSimulMCMC(iterMCMC)
            self.output(histHyp=histHyp.T, histF=histF)

            # idx_100 = range(iterMCMC*3/4-1, iterMCMC, 100)
            # ll = np.mean(histHyp[0, idx_100])
            # sf = np.mean(histHyp[1, idx_100])
            # sn = np.mean(histHyp[2, idx_100])
            #
            # covMCMC = kcGP.covK.RBF(np.log(ll), np.log(sf))
            # model.setPrior(kernel=covMCMC)
            # model.setNoise(np.log(sn))
            # model.getPosterior(self.x, self.y)

class crossValid(Framework):
    """Second framework: Gaussian process with cross validation

    Parameters
    ----------
    data: ndarray with shape (n_samples, n_features+1)
        target & training samples
    window: integer
        width of window to scan through the data
    gap: array
        interval of observations
    """
    def __init__(self, data, window, gapArray):
        super(crossValid, self).__init__(data, window, gapArray)
        self.name = 'Second framework: GP with window width: %d' %(window)
        self.windowSize = window
        self.gapArray = gapArray

    def execute(self, updOpt=None, iterMCMC=1000):
        self.getScaleData()

        originalX = self.x[:]
        originalY = self.y[:]

        for gap in self.gapArray:
            gapLLK = []
            numFold = gap+self.windowSize

            for fold in range(numFold):
                print 'Gap: %r, fold %d' %(gap, fold+1)
                foldLLK = []
                model = kcGP.gpK.GPR()

                self.x, self.y = originalX, originalY
                self.x, self.y, valX, valY, _ = self.getFoldData(fold, gap, window=self.windowSize)

                assert updOpt is not None, 'Please choose either optimization or MCMC'
                if updOpt == 'opt':
                    model.optimize(self.x, self.y)

                elif updOpt == 'mcmcSml':
                    foldF, foldHyp = self.runSimulMCMC(iterMCMC)

                model.xs = valX
                upper = 100. - np.mean(self.y)
                lower = 0. - np.mean(self.y)
                selected_sample = range(iterMCMC*9/10-1, iterMCMC, 10)
                for i in selected_sample:
                    ll = foldHyp[0, i]
                    sf = foldHyp[1, i]
                    sn = foldHyp[2, i]
                    propF = foldF[:, i].reshape((foldF.shape[0], 1))

                    covMCMC = kcGP.covK.RBF(np.log(ll), np.log(sf))
                    model.setPrior(kernel=covMCMC)
                    model.setNoise(np.log(sn))
                    model.setData(self.x, self.y)

                    trunclik = kcGP.likK.TruncatedGauss2(upper=upper, lower=lower, log_sigma=np.log(sn))
                    model.likfunc = trunclik

                    # inference fs|f
                    ys, _, _, fs2 = sdsK.inf_mcmc(propF, model)

                    trunclik.upper = 100.
                    trunclik.lower = 0.
                    foldLLK.append(trunclik.evaluate(y=ys, mu=valY, s2=fs2)/ys.shape[0])

                gapLLK.append(np.mean(foldLLK))

            # self.testLLK[gap] = gapLLK
            self.output(gap, foldHyp.T, foldF, gapLLK)

class autoregressive(Framework):
    """Baseline framework: Autoregressive model
    """
    def __init__(self, data, window, gapArray, lag):
        super(autoregressive, self).__init__(data, window, gapArray)
        self.name = 'Comparison: AR(1)'
        self.windowSize = window
        self.gapArray = gapArray
        self.lag = lag

    def execute(self, updOpt=None, iterMCMC=None):
        self.getScaleData()

        gaussianLik = kcGP.likK.Gauss(log_sigma=np.log(1.2))

        for gap in self.gapArray:
            gapLLK = []
            numFold = gap+self.windowSize

            for fold in range(numFold):
                print 'Gap: %r, fold %d' %(gap, fold+1)
                _, _, _, valY, valIdx = self.getFoldData(fold, gap, window=self.windowSize)
                valIdx -= 1
                if -1 in valIdx:
                    valIdx = valIdx[valIdx>=0]
                    valY = valY[1:]

                model = sm.tsa.AR(self.y)
                model_res = model.fit(maxlag=self.lag)
                y_pred = model_res.fittedvalues         # y_0 has no fitted value

                foldLLK = gaussianLik.evaluate(y_pred[valIdx], mu=valY)/np.shape(valY)[0]
                gapLLK.append(foldLLK)

            self.output(gap, llk=gapLLK)

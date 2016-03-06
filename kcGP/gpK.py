'''
Created on Jun 25, 2015
@Copyright by Marion Neumann and Shan Huang, 30/09/2013
http://www-ai.cs.uni-dortmund.de/weblab/static/api_docs/pyGPs/
'''
import numpy as np
import meanK, covK, likK, infK
import matplotlib.pyplot as plt
from copy import deepcopy
from tools import unique, jitchol
# from pyGPs.Core import opt

class GPK(object):
    '''
    Base class of Gaussian Processes
    '''

    def __init__(self):
        '''
        Constructor
        '''
        super(GPK, self).__init__()
        self.usingDefaultMean = True  # was using default mean function now?
        self.meanfunc = None      # mean function
        self.covfunc = None       # covariance function
        self.likfunc = None       # likelihood function
        self.inffunc = None       # inference function
        self.optimizer = None     # optimizer object
        self.nlZ = None           # negative log marginal likelihood
        self.dnlZ = None          # column vector of partial derivatives of the negative log marginal likelihood w.r.t. each hyperparameter
        self.posterior = None     # struct representation of the (approximate) posterior
        self.x = None             # n by D matrix of training inputs
        self.y = None             # column vector of length n of training targets
        self.xs = None            # n by D matrix of test inputs
        self.ys = None            # column vector of length nn of true test targets (optional)
        self.ym = None            # column vector (of length ns) of predictive output means
        self.ys2 = None           # column vector (of length ns) of predictive output variances
        self.fm = None            # column vector (of length ns) of predictive latent means
        self.fs2 = None           # column vector (of length ns) of predictive latent variances
        self.lp = None            # column vector (of length ns) of log predictive probabilities

    def setData(self, x, y):
        '''
        Set training inputs and training labels to model.

        :param x: training inputs in shape (n,D)
        :param y: training labels in shape (n,1)
        '''
        # check whether the number of inputs and labels match
        assert x.shape[0] == y.shape[0], 'number of inputs and labels does not match'

        # check the shape of inputs
        # transform to the correct shape
        if x.ndim == 1:
            x = np.reshape(x, (x.shape[0],1))
        if y.ndim == 1:
            y = np.reshape(y, (y.shape[0],1))

        self.x = x
        self.y = y
        if self.usingDefaultMean:
            c = np.mean(y)
            self.meanfunc = meanK.Const(c)   # adapt default prior meanK wrt. training labels

    def setPrior(self, mean=None, kernel=None):
        '''
        Set prior meanK and covariance other than the default setting of current model.

        :param meanK: instance of meanK class
        :param kernel: instance of covariance class
        '''
        if not mean is None:
            assert isinstance(mean, meanK.Mean), 'meanK function is not an instance of meanK.Mean class'
            self.meanfunc = mean
            self.usingDefaultMean = False
        if not kernel is None:
            assert isinstance(kernel, covK.Kernel), 'covK function is not an instance of covK.Kernel class'
            self.covfunc = kernel
            if type(kernel) is covK.Pre:
                self.usingDefaultMean = False

    def setOptimizer(self, method, num_restarts=None, min_threshold=None, meanRange=None, covRange=None, likRange=None):
        '''
        This method is used to specify optimization configuration.
        By default, gp uses a single run "minimize".
        :param method: Optimization methods. Possible values are:\n
                       "Minimize"   -> minimize by Carl Rasmussen (python implementation of "minimize" in GPML)\n
                       "CG"         -> conjugent gradient\n
                       "BFGS"       -> quasi-Newton method of Broyden, Fletcher, Goldfarb, and Shanno (BFGS)\n
                       "SCG"        -> scaled conjugent gradient (faster than CG)\n
        :param num_restarts: Set if you want to run mulitiple times of optimization with different initial guess.
                             It specifys the maximum number of runs/restarts/trials.
        :param min_threshold: Set if you want to run mulitiple times of optimization with different initial guess.
                              It specifys the threshold of objective function value. Stop optimization when this value is reached.
        :param meanRange: The range of initial guess for mean hyperparameters.
                          e.g. meanRange = [(-2,2), (-5,5), (0,1)].
                          Each tuple specifys the range (low, high) of this hyperparameter,
                          This is only the range of initial guess, during optimization process, optimal hyperparameters may go out of this range.
                          (-5,5) for each hyperparameter by default.
        :param covRange: The range of initial guess for kernel hyperparameters. Usage see meanRange
        :param likRange: The range of initial guess for likelihood hyperparameters. Usage see meanRange
        '''
        pass

    def optimize(self, x=None, y=None, numIterations=40):
        '''
        Train optimal hyperparameters based on training data,
        adjust new hyperparameters to all mean/cov/lik functions.
        :param x: training inputs in shape (n,D)
        :param y: training labels in shape (n,1)
        '''
        # check whether the number of inputs and labels match
        if x is not None and y is not None:
            assert x.shape[0] == y.shape[0], "number of inputs and labels does not match"

        # check the shape of inputs
        # transform to the correct shape
        if not x is None:
            if x.ndim == 1:
                x = np.reshape(x, (x.shape[0],1))
            self.x = x

        if not y is None:
            if y.ndim == 1:
                y = np.reshape(y, (y.shape[0],1))
            self.y = y

        if self.usingDefaultMean and self.meanfunc is None:
            c = np.mean(y)
            self.meanfunc = meanK.Const(c)    # adapt default prior mean wrt. training labels

        # optimize
        optimalHyp, optimalNlZ = self.optimizer.findMin(self.x, self.y, numIters = numIterations)
        self.nlZ = optimalNlZ

        # apply optimal hyp to all mean/cov/lik functions here
        self.optimizer._apply_in_objects(optimalHyp)
        self.getPosterior()

    def getPosterior(self, x=None, y=None, der=True):
        '''
        Fit the training data. Update negative log marginal likelihood(nlZ),
        partial derivatives of nlZ w.r.t. each hyperparameter(dnlZ),
        and struct representation of the (approximate) posterior(post),
        which consists of post.alpha, post.L, post.sW.

        nlZ, dnlZ, post = getPosterior(x, y, der=True)\n
        nlZ, post       = getPosterior(x, y, der=False )

        :param x: training inputs in shape (n,D)
        :param y: training labels in shape (n,1)
        :param boolean der: flag for whether to compute derivatives

        :return: negative log marginal likelihood (nlZ), derivatives of nlZ (dnlZ), posterior structure(post)
        '''
        # check whether the number of inputs and labels match
        if x is not None and y is not None:
            assert x.shape[0] == y.shape[0], "number of inputs and labels does not match"

        # check the shape of inputs
        # transform to the correct shape
        if not x is None:
            if x.ndim == 1:
                x = np.reshape(x, (x.shape[0],1))
            self.x = x

        if not y is None:
            if y.ndim == 1:
                y = np.reshape(y, (y.shape[0],1))
            self.y = y

        if self.usingDefaultMean and self.meanfunc is None:
            c = np.mean(y)
            self.meanfunc = meanK.Const(c)    # adapt default prior mean wrt. training labels

        # call inference method
        if isinstance(self.likfunc, likK.Erf):
            uy = unique(self.y)
            ind = (uy != 1)
            if any(uy[ind] != -1):
                    raise Exception('You attempt classification using labels different from {+1,-1}')

        if not der:
            post, nlZ = self.inffunc.evaluate(self.meanfunc, self.covfunc, self.likfunc, self.x, self.y, 2)
            self.nlZ = nlZ
            self.posterior = deepcopy(post)

            return nlZ, post

        else:
            post, nlZ, dnlZ = self.inffunc.evaluate(self.meanfunc, self.covfunc, self.likfunc, self.x, self.y, 3)
            self.nlZ = nlZ
            self.dnlZ = deepcopy(dnlZ)
            self.posterior = deepcopy(post)

            return nlZ, dnlZ, post

    def predict(self, xs, ys=None):
        '''
        Prediction of test points (given by xs) based on training data of the current model.
        This method will output the following value:\n
        predictive output means(ym),\n
        predictive output variances(ys2),\n
        predictive latent means(fm),\n
        predictive latent variances(fs2),\n
        log predictive probabilities(lp).\n
        Theses values can also be achieved from model's property. (e.g. model.ym)
        :param xs: test input in shape of nn by D
        :param ys: test target(optional) in shape of nn by 1 if given
        :return: ym, ys2, fm, fs2, lp
        '''
        # check the shape of inputs
        # transform to correct shape if neccessary
        if xs.ndim == 1:
            xs = np.reshape(xs, (xs.shape[0],1))
        self.xs = xs
        if not ys is None:
            if ys.ndim == 1:
                ys = np.reshape(ys, (ys.shape[0],1))
            self.ys = ys

        meanfunc = self.meanfunc
        covfunc  = self.covfunc
        likfunc  = self.likfunc
        inffunc  = self.inffunc
        x = self.x
        y = self.y
        my = np.mean(y)

        if self.posterior is None:
            self.getPosterior()
        alpha = self.posterior.alpha
        L     = self.posterior.L
        sW    = self.posterior.sW

        nz = range(len(alpha[:,0]))         # non-sparse representation
        if L == []:                         # in case L is not provided, we compute it
            K = covfunc.getCovMatrix(x=x[nz,:], mode='train')
            #L = np.linalg.cholesky( (np.eye(nz) + np.dot(sW,sW.T)*K).T )
            L = jitchol( (np.eye(len(nz)) + np.dot(sW,sW.T)*K).T )
        Ltril     = np.all( np.tril(L,-1) == 0 ) # is L an upper triangular matrix?
        ns        = xs.shape[0]                  # number of data points
        nperbatch = 1000                         # number of data points per mini batch
        nact      = 0                            # number of already processed test data points
        ymu = np.zeros((ns,1))
        ys_up = np.zeros((ns,1))
        ys_lw = np.zeros((ns,1))
        fmu = np.zeros((ns,1))
        fs2 = np.zeros((ns,1))
        lp  = np.zeros((ns,1))
        while nact<=ns-1:                              # process minibatches of test cases to save memory
            id  = range(nact,min(nact+nperbatch,ns))   # data points to process
            kss = covfunc.getCovMatrix(z=xs[id,:], mode='self_test')  # self-variances
            Ks  = covfunc.getCovMatrix(x=x[nz,:], z=xs[id,:], mode='cross')   # cross-covariances
            ms  = meanfunc.getMean(xs[id,:])
            N   = (alpha.shape)[1]                     # number of alphas (usually 1; more in case of sampling)
            Fmu = np.tile(ms,(1,N)) + np.dot(Ks.T,alpha[nz])          # conditional mean fs|f
            fmu[id] = np.reshape(Fmu.sum(axis=1)/N,(len(id),1))       # predictive means

            if Ltril: # L is triangular => use Cholesky parameters (alpha,sW,L)
                V       = np.linalg.solve(L.T,np.tile(sW,(1,len(id)))*Ks)
                fs2[id] = kss - np.array([(V*V).sum(axis=0)]).T             # predictive variances
            else:     # L is not triangular => use alternative parametrization
                fs2[id] = kss + np.array([(Ks*np.dot(L,Ks)).sum(axis=0)]).T # predictive variances
            fs2[id] = np.maximum(fs2[id],0)            # remove numerical noise i.e. negative variances
            Fs2 = np.tile(fs2[id],(1,N))               # we have multiple values in case of sampling
            if ys is None:
                trunclik = likK.TruncatedGauss(100.-my, 0.-my, likfunc.hyp[0])
                lp, Ymu, Lower, Upper = trunclik.evaluate(None, Fmu[:], Fs2[:], None, None, 3)
            else:
                Lp, Ymu, Ys2 = likfunc.evaluate(np.tile(ys[id],(1,N)), Fmu[:], Fs2[:],None,None,3)
            # lp[id]  = np.reshape( np.reshape(Lp,(np.prod(Lp.shape),N)).sum(axis=1)/N , (len(id),1) )   # log probability; sample averaging
            ymu[id] = np.reshape( np.reshape(Ymu,(np.prod(Ymu.shape),N)).sum(axis=1)/N ,(len(id),1) )  # predictive mean ys|y and ...
            ys_up[id]=np.reshape( np.reshape(Upper,(np.prod(Upper.shape),N)).sum(axis=1)/N, (len(id), 1))
            ys_lw[id]=np.reshape( np.reshape(Lower,(np.prod(Lower.shape),N)).sum(axis=1)/N, (len(id), 1))
            # ys2[id] = np.reshape( np.reshape(Ys2,(np.prod(Ys2.shape),N)).sum(axis=1)/N , (len(id),1) ) # .. variance
            nact = id[-1]+1                  # set counter to index of next data point
        self.ym = ymu + my
        # self.ys2 = ys2
        self.lp = lp
        self.fm = fmu
        self.fs2 = fs2
        ys_lw = np.reshape(np.mean(ys_lw, axis=1), (ns, 1))
        ys_up = np.reshape(np.mean(ys_up, axis=1), (ns, 1))

        if ys is None:
            return ymu, ys_lw, ys_up, fmu, fs2, None
        else:
            return ymu, None, fmu, fs2, lp

class GPR(GPK):
    '''
    Gaussian process for regression
    '''
    def __init__(self):
        super(GPR, self).__init__()
        self.meanfunc = meanK.Zero()
        self.covfunc = covK.RBF()
        self.likfunc = likK.Gauss()
        self.inffunc = infK.Exact()
        self.optimizer = None #opt.Minimize(self)

    def setNoise(self, log_sigma):
        '''
        Set noise other than default noise value
        :param log_sigma: logorithm of the noise sigma
        '''
        self.likfunc = likK.Gauss(log_sigma)

    def plot(self, ys_lw, ys_up, f=None, fs=None):
        '''
        Plot 1d GP regression result.
        :param list axisvals: [min_x, max_x, min_y, max_y] setting the plot range
        '''
        xs = self.xs
        x = self.x
        y = self.y
        ym = self.ym
        # ym = np.minimum(np.maximum(self.ym, 0.), 4.6)
        ys2 = self.ys2
        SHADEDCOLOR = [0.7539, 0.89453125, 0.62890625, 1.0]
        MEANCOLOR = [ 0.2109375, 0.63385, 0.1796875, 1.0]
        DATACOLOR = [0.12109375, 0.46875, 1., 1.0]

        plt.figure()
        xss  = np.reshape(xs, (xs.shape[0],))
        ymm  = np.reshape(ym, (ym.shape[0],))
        ys_lw= np.reshape(ys_lw, (ys_lw.shape[0],))
        ys_up= np.reshape(ys_up, (ys_up.shape[0],))
        # ys22 = np.reshape(ys2,(ys2.shape[0],))

        plt.plot(x, y, color=DATACOLOR, ls='None', marker='+', ms=8, mew=2)
        plt.plot(xss, ymm, color=MEANCOLOR, ls='-', lw=2.5)
        plt.fill_between(xss, ys_up, ys_lw, facecolor=SHADEDCOLOR, linewidths=0.0)
        plt.grid()

        # if not (f is None):
        #     f = f[:, 0:20]
        #     x = np.tile(x, (1, f.shape[1]))
        #     colors = iter(cm.rainbow(np.linspace(0, 1, f.shape[1])))
        #
        #     for i in range(f.shape[1]):
        #         plt.scatter(x[:, i], f[:, i], color=next(colors))#, ls='None', marker='o', ms=3.5, mew=2)
        #
        # if not (fs is None):
        #     plt.plot(xs, fs, color=np.arange(fs.shape[1]), ls='None', marker='o', ms=3.5, mew=2)

        plt.xlabel('input x')
        plt.ylabel('target y')
        plt.xlim(xmin=np.min(x)-1, xmax=np.max(x)+1)
        plt.ylim(ymin=np.min(y)-1, ymax=np.max(y)+1)
        plt.show()
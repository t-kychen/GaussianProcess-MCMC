'''
Created on Jul 5, 2015

Implementation of MCMC sampling:
    1) Metropolis-Hastings
    2) Slice sampling
    3) Elliptical slice sampling

@author: Thomas
'''

import joint_dist
import numpy as np
import math
import matplotlib.pyplot as plt
from kcGP import covK, likK, tools
from pyhmc import hmc


def metropolis(init, iters):
    '''
    Based on http://www.cs.toronto.edu/~asamir/cifar/rpa-tutorial.pdf
    '''
    dist = joint_dist.Joint_dist()
    
    # set up empty sample holder
    D = len(init)
    samples = np.zeros((D, iters))
    
    # initialize state and log-Likelihood
    state = init.copy()
    Lp_state = dist.loglike(state)
    
    accepts = 0.
    for i in np.arange(0, iters):
        
        # propose a new state, require by the web assignment
        prop = np.random.multivariate_normal(state.ravel(), np.eye(10)).reshape(D,1)
        
        Lp_prop = dist.loglike(prop)
        rand = np.random.rand()
        if np.log(rand) < (Lp_prop - Lp_state):
            accepts += 1
            state = prop.copy()
            Lp_state = Lp_prop
            
        samples[:, i] = state.copy().ravel()
        
    print 'Acceptance ratio', float(accepts/iters)
    return samples


def slice_sampling(init, iters, sigma, step_out=True):
    '''
    Based on http://homepages.inf.ed.ac.uk/imurray2/teaching/09mlss/
    '''
    dist = joint_dist.Joint_dist()
    
    # set up empty sample holder
    D = len(init)
    samples = np.zeros((D, iters))
    
    # initialize
    xx = init.copy()
    
    for i in xrange(iters):
        perm = range(D)
        np.random.shuffle(perm)
        last_llh = dist.loglike(xx)
        
        for d in perm:
            llh0 = last_llh + np.log(np.random.rand())      # np.random.rand() randomly generates a number between 0 and 1
            rr = np.random.rand(1)                          # np.random.rand(1) does the same thing above, but puts it in an array
            # l and r here stand for left and right
            x_l = xx.copy()
            x_l[d] = x_l[d] - rr * sigma[d]
            x_r = xx.copy()
            x_r[d] = x_r[d] + (1 - rr) * sigma[d]
            
            if step_out:
                llh_l = dist.loglike(x_l)
                while llh_l > llh0:
                    x_l[d] = x_l[d] - sigma[d]
                    llh_l = dist.loglike(x_l)
                llh_r = dist.loglike(x_r)
                while llh_r > llh0:
                    x_r[d] = x_r[d] + sigma[d]
                    llh_r = dist.loglike(x_r)
            
            x_cur = xx.copy()
            while True:
                xd = np.random.rand() * (x_r[d] - x_l[d]) + x_l[d]
                x_cur[d] = xd.copy()
                last_llh = dist.loglike(x_cur)
                if last_llh > llh0:
                    xx[d] = xd.copy()
                    break
                elif xd > xx[d]:
                    x_r[d] = xd
                elif xd < xx[d]:
                    x_l[d] = xd
                else:
                    raise RuntimeError('Slice sampling shrank too far.')
        
        if i % 1000 == 0: print 'iteration', i
        
        samples[:, i] = xx.copy().ravel()
        
    return samples


def elliptical_slice(var):
    '''
    Elliptical slice sampling
    
    :param f: initial latent variable
    :param y: input y
    :param prior: cholesky decomposition of the covariance matrix based on x
    
    :return ll: log-likelihood
    :return ff: latent variable
    '''
    f = var[0]
    x = var[1]
    y = var[2]
    hyp = var[3]
    opt = var[4]

    n = f.shape[0]    
    
    # First, compute the log likelihood of the initial state i.e. log L(f)
    cur_logf, junk = logGP(x, f, y, hyp, option='ell')
    #llk = likK.Gauss()
    #cur_logf = np.sum(llk.evaluate(f, y, nargout=1))
        
    # Set up the ellipse (nu) and the slice threshold (log_y, or log L(y))
    Kc    = covK.RBF(np.log(hyp[0]), np.log(hyp[1]))
    K     = Kc.getCovMatrix(x=x, mode='train')
    prior = tools.jitchol(K+np.eye(n)).T
    if not prior.shape[0] == n or not prior.shape[1] == n:
        raise IOError('Prior must be given by a n-element sample or nxn chol(Sigma)')
    n_nu = np.dot(prior, np.random.normal(size=n))      # D*1 = (D*D) * (D*1)
    nu = n_nu.reshape((n,1))        
    
    log_y = cur_logf + math.log(np.random.uniform())
    
    # Set up a bracket of angles
    phi = np.random.uniform(high=2.*math.pi)
    phi_min = phi - 2. * math.pi
    phi_max = phi
    
    # Slice sampling loop
    while True:  
        # Compute log-likelihood for proposed latent variable and check if it's on the slice
        prop_f = f * math.cos(phi) + nu * math.sin(phi)
        prop_logf, junk = logGP(x, prop_f, y, hyp, option='ell')
        #propDist  = likK.Gauss()
        #prop_logf = np.sum(propDist.evaluate(prop_f, y, nargout=1))

        if prop_logf > log_y:                
            # Proposed point is on the slice, ACCEPT IT
            return prop_logf, prop_f

        else:
            # Shrink the bracket and try a new point
            if phi >= 0:       
                phi_max = phi            
            else:
                phi_min = phi
            
            # Try a new point
            phi = np.random.uniform(low=phi_min, high=phi_max)


def surrogate_slice_sampling(f, x, y, sigma, model, option):
    '''
    Surrogate data slice sampling
    
    :param f: initial latent variable
    :param x: input x
    :param y: input y
    :param sigma: scale
    :param model: GP model
    :param option: which hyperparameter to be updated
    
    :return prop_f: proposal latent variable
    :return prop_hyp: proposal hyperparameters
    '''
    K = model.covfunc.getCovMatrix(x=x, mode='train')
    S = 1.0         # auxiliary nose
    '''
    ***update S
    '''
    if option == 'ell':
        hyp = np.exp(model.covfunc.hyp[0])
    elif option == 'sf2':
        hyp = np.exp(2.*model.covfunc.hyp[1])

    # draw surrogate data g from N(f, S_theta) or N(0, Cov + S_theta)
    m_theta_g, chol_R_theta = aux_var_model(f, K, S)
    # compute implied latent variables
    n = np.linalg.inv(chol_R_theta) * (f - m_theta_g)
    
    # randomly center a bracket v
    v = np.random.uniform() * sigma
    theta_min = hyp - v
    theta_max = theta_min + sigma
    
    # draw u from Uniform(0,1)
    u = np.random.uniform()
    
    # determine threshold
    initDist = likK.Gauss()
    '''
    ***update N(g; 0, Cov + S_theta)
    ***update prior of hyp -gamma
    '''
    y = u * initDist.evaluate(f,y) * np.random.normal() * np.random.gamma()
    
    while True:
        # draw proposal hyp
        prop_hyp = np.random.uniform(low=theta_min, high=theta_max)

        # compute proposal latent variables
        if option == 'ell':
            model.covfunc.hyp[0] = np.log(prop_hyp)
        elif option == 'sf2':
            model.covfunc.hyp[1] = np.log(prop_hyp)/2.
        K = model.covfunc.getCovMatrix(x=x, mode='train')
        '''
        ***update S
        '''
        m_theta_g, chol_R_theta = aux_var_model(f, K, S)        
        prop_f = np.dot(chol_R_theta, n) + m_theta_g
    
        propDist = likK.Gauss()
        '''
        ***update N(g; 0, Cov + S_theta)
        ***update prior of hyp -gamma
        '''
        if propDist.evaluate(prop_f, y) * np.random.normal() * np.random.gamma() > y:
            return prop_f, prop_hyp
        elif prop_hyp < hyp:
            theta_min = prop_hyp
        else:
            theta_max = prop_hyp


def aux_var_model(ll, K, S):
    '''
    auxiliary variable model introducing surrogate Gaussian observations
    
    @param ll: latent variables
    @param K: covariance
    @param S: auxiliary noise
    '''
    n = K.shape[0]
    S_theta = np.eye(n) * S                                                 # size = n*n
    g = np.dot(np.linalg.cholesky((K + S_theta)), np.random.normal(size=n)) # size = n*1
    
    R_theta   = np.linalg.inv(np.linalg.inv(K) + np.linalg.inv(S_theta))    # size = n*n
    m_theta_g = R_theta * np.linalg.inv(S_theta) * g                        # size = n*1
    L_R_theta = np.linalg.cholesky(R_theta)
    
    return m_theta_g, L_R_theta

    
def hmcK(x, E, var, leapfrog, epsilon, nsamples):
    '''
    try to implement hamiltonian monte carlo
    
    @param x: initial state
    @param E: function E(x) and its gradient
    @param var: parameters for computing log-likelihood and its gradient
    @param leapfrog: number of steps in leapfrog
    @param epsilon: step size in leapfrog
    @param nsamples: number of samples to be drawn
    
    @return new state, new llk
    '''
    log, grad = E(x, var)      
    # E = -llk
    log = 0 - log
    grad = 0 - grad        
    
    while nsamples > 0:
        p = np.random.randn()                                # initialize momentum with Normal(0,1)
        H = np.dot(p,p) / 2. + log                           # compute the current hamiltonian function

        x_new = x
        grad_new = grad
        
        for tau in range(leapfrog):
            
            p = p - epsilon * grad_new / 2.                  # make half-step in p
            x_new = x_new + epsilon * p                      # make full step in x
            
            log_new, grad_new = E(x_new, var)                # find new gradient
            log_new = 0 - log_new
            grad_new = 0 - grad_new
            
            p = p - epsilon * grad_new / 2.
                
        H_new = np.dot(p,p) / 2. + log_new                   # compute new hamiltonian function
        delta_H = H_new - H                                  # decide whether to accept

        if delta_H < 0 or np.random.rand() < np.exp(-delta_H):
            print 'accept!'
            grad = grad_new
            x = x_new
            log = log_new
            
        else:
            print 'reject!'
        
        nsamples -= 1

    return x, log

            
def logp_hyper(state, var):
    '''
    log likelihood of hyperparameters and their gradients
    
    @param state: input state for HMC
    @param var: [latent variables (f), input x (x), input y (y), hyp (hyperparameter), option (opt)]
    @return: log probability, gradient
    '''
    f = var[0]
    x = var[1]
    y = var[2]
    hyp = var[3]
    opt = var[4]
    
    if opt == 'ell':
        gamma = [0.1, 10]
        hyp[0] = state
    elif opt == 'sf2':
        gamma = [0.1, 10]
        hyp[1] = state
    elif opt == 'noise':
        gamma = [0.1, 10]
        hyp[2] = state
        
    # llk of GP distribution
    logN, gradN = logGP(x, f, y, hyp, opt)
    
    # llk of gamma distribution
    logG, gradG = logGamma(state, k=gamma[0], theta=gamma[1])
    
    #print 'logN: ', logN, ' logG: ', logG
    logp = logN + logG
    grad = gradN + gradG
    
    return logp, grad


def logGP(x, f, y, hyp, option):
    '''
    calculate log-likelihood of GP distribution and its gradient
    i.e. log p( f | hyper )
    
    @param x: observation x
    @param f: latent variables
    @param y: observation y
    @param hyp: hyper in covariance
    @param K: covariance function of x
    @param option: which hyperparamter to be used for derivatives, must be ell, sf2, or noise
    
    @return: log-likelihood of normal distribution, its gradient w.r.t. hyperparameters
    '''
    sf2 = np.exp(2.*hyp[1])                                 # hyperparameter (sigma_y)^2 in RBF kernel (covariance function)
    sn2 = np.exp(2.*hyp[2])                                 # noise (sigma_n)^2
        
    # covariance matrix
    covCur= covK.RBF(np.log(hyp[0]), np.log(hyp[1]))
    K     = covCur.getCovMatrix(x=x, mode='train')
    n     = np.shape(x)[0]
    L     = tools.jitchol(K+np.eye(n)).T
    alpha = tools.solve_chol(L,f)
    if option in ['ell', 'sf2']:
        # log likelihood
        logN  = -(0.5*np.dot(f.T,alpha) + np.log(np.diag(L)).sum() + 0.5*n*np.log(2*np.pi))
        
        # gradient of llk
        Q = tools.solve_chol(L,np.eye(n)) - np.dot(alpha,alpha.T)        # precompute for convenience
        if option == 'ell':                                              # compute derivative matrix w.r.t. 1st parameter
            derK = sf2 * np.exp(-0.5*K) * K
        elif option == 'sf2':                                            # compute derivative matrix w.r.t. 2nd parameter
            derK = 2. * sf2 * np.exp(-0.5*K)
         
        gradN = (Q*derK).sum()/2.
        
        return logN.sum(), gradN                          

    elif option == 'noise':
        # log likelihood
        logN  = np.sum(-(y-f)**2 / sn2/2 - np.log(2.*np.pi*sn2)/2.)
            
        # gradient of llk
        gradN = np.sum((y-f)**2 * sn2**(-3/2) - sn2**(-1/2))             # compute derivative matrix w.r.t. noise
            
        return logN, gradN


def logGamma(state, k, theta):
    '''
    calculate log-likelihood of gamma distribution and the gradient of it
    
    @param x: hyperparameter to be updated
    @param k: shape parameter of gamma distribution
    @param theta: scale parameter of gamma distribution
    
    @return: log-likelihood of gamma distribution, its gradient w.r.t. hyperparameters
    '''
    # log-likelihood
    logG = (k-1)*np.log(state) - state/theta - k*np.log(theta) - np.log(math.gamma(k))
    
    # gradient of llk
    gradG = (k-1)*(1/state) - 1/theta

    return logG, gradG


def plotMCMC(input_x, input_y, iter, input_xs=None, input_ym=None, input_ys=None):
    '''
    Visualizing the result of sampling
    
    :param input_x:  training data x
    :param input_y:  training data y
    :param iter:     whether to plot the log-likelihood or visualize the result
    :param input_xs: testing data xs
    :param input_ym: predicted mean of y
    :param input_ys: predicted variance of y
    '''
    SHADEDCOLOR = [0.7539, 0.89453125, 0.62890625, 1.0]
    MEANCOLOR = [ 0.2109375, 0.63385, 0.1796875, 1.0]
    DATACOLOR = [0.12109375, 0.46875, 1., 1.0]
    
    plt.figure()
    if iter:
        plt.plot(input_x, input_y, color=DATACOLOR, ls='-', lw=3.)
        plt.xlabel('Iter')
        plt.ylabel('Log likelihood')
    else:
        xss  = np.reshape(input_xs,(input_xs.shape[0],))
        ymm  = np.reshape(input_ym,(input_ym.shape[0],))
        ys22 = np.reshape(input_ys,(input_ys.shape[0],))
        
        plt.plot(input_x, input_y, color=DATACOLOR, ls='None', marker='+',ms=12, mew=2)
        plt.plot(input_xs, input_ym, color=MEANCOLOR, ls='-', lw=3.)
        plt.fill_between(xss, ymm + 2.*np.sqrt(ys22), ymm - 2.*np.sqrt(ys22), facecolor=SHADEDCOLOR, linewidths=0.0)
        plt.grid()
        
        plt.xlabel('x: Distance from the Beginning Point (mile)')
        plt.ylabel('y: Proposed Latent Variable f')        

    plt.show()


def predictMCMC(x, y, xs, propF, L, meanfunc, covfunc, likfunc, ys=None):
    '''
    :param xs: test input in shape of nn by D

    :return: ym, ys2, fm, fs2

    '''
    alpha = tools.solve_chol(L,propF).reshape((np.shape(x)[0],1))/np.exp(2*likfunc.hyp[0])
    sW = np.ones((np.shape(x)[0],1))/np.sqrt(np.exp(2. * likfunc.hyp[0]))
    Ltril     = np.all( np.tril(L,-1) == 0 ) # is L an upper triangular matrix?
    
    kss = covfunc.getCovMatrix(z=xs, mode='self_test')    # self-variances
    ks  = covfunc.getCovMatrix(x=x, z=xs, mode='cross')   # cross-covariances
    ms  = meanfunc.getMean(xs)
    Fmu = ms + np.dot(ks.T,alpha)          # conditional mean fs|f
            
    if Ltril: # L is triangular => use Cholesky parameters (alpha,sW,L)
        V   = np.linalg.solve(L.T,sW*ks)
        fs2 = kss - np.array([(V*V).sum(axis=0)]).T             # predictive variances
    else:     # L is not triangular => use alternative parametrization
        fs2 = kss + np.array([(ks*np.dot(L,ks)).sum(axis=0)]).T # predictive variances
    fs2 = np.maximum(fs2,0)            # remove numerical noise i.e. negative variances
    Fs2 = fs2                          # we have multiple values in case of sampling
    
    predictedLLK = likK.Gauss()
    Lp, Ymu, Ys2 = predictedLLK.evaluate(ys,Fmu[:],Fs2[:],None,None,3)

        
    return Lp, Ymu, Ys2
    

if __name__ == '__main__':
    '''
    from matplotlib import rcParams
    import prettyplotlib as pplt
    import matplotlib.pyplot as plt
    
    rcParams['font.size'] = 18
    rcParams['figure.figsize'] = (10, 6)
    
    # define the starting point
    w_0 = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    
    '''
    '''
    # actually do the sampling
    n = 50000
    print 'Sampling method: metropolis-hastings'
    samples = metropolis(w_0, n)
    
    n = 10000
    sigma = np.ones(10)
    print 'Sampling method: slice sampling'
    samples = slice_sampling(w_0, iters=n, sigma=sigma)
    n = 10000
    print 'Sampling method: elliptical slice sampling'
    mat = np.zeros((10,1000))
    for d in range(10):
        mat[d,:] = np.random.normal(0,np.sqrt(np.e),1000)
    cov_mat = np.cov(mat)
    prior = np.linalg.cholesky(cov_mat)
    junk, samples = elliptical_slice(w_0, prior, iters=n)
    
    print samples.shape
    v = samples[0, :]
    fig, (ax0,ax1) = plt.subplots(2,1)
    
    # show values of sampled v by iteration
    pplt.plot(ax0, np.arange(n), v)
    ax0.set_xlabel('iteration number')
    ax0.set_ylabel('values of sampled v')
    
    # plot a histogram of values of v
    pplt.hist(ax1, v, bins=80)
    ax1.set_xlabel('values of sampled v')
    ax1.set_ylabel('observations')
    
    fig.savefig('MCMC_example.png')
    '''
    
    # test case
    import triangle
    def logprob(x, ivar):
        logp = -0.5 * np.sum(ivar * x**2)
        grad = -ivar * x
        
        return logp, grad
    
    ivar = 1. / np.random.rand(5)
    '''
    samples = np.zeros((20000,5))
    for s in range(np.shape(samples)[0]):
        samples[s,:], junk = hmcK(q=np.random.randn(5), E=logprob, ivar=ivar, leapfrog=1, epsilon=0.2)
    '''
    samples, logp = hmc(logprob, x0=np.random.randn(5), args=(ivar,), n_samples=1e4, return_logp=True)
    print samples[-100:-1,0]
    figure = triangle.corner(samples)
    figure.savefig('HMC0825.png')
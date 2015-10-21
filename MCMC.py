'''
Created on Jul 5, 2015

Implementation of MCMC sampling:
    1) Elliptical slice sampling
    2) Surrogate data slice sampling
    3) Hybrid Monte Carlo

@author: Thomas
'''
import numpy as np
import scipy.spatial.distance as spdist
import scipy.special
from scipy.stats import norm, multivariate_normal, gamma
from kcGP import covK, likK, tools
from pyhmc import hmc


def elliptical_slice(var, sn):
    '''
    Elliptical slice sampling
    
    :param f: initial latent variable
    :param x: input x
    :param y: input y
    :param sn: parameter in likelihood func

    :return ll: log-likelihood
    :return ff: latent variable
    '''
    f = var[0]
    x = var[1]
    y = var[2]
    hyp = var[3]

    n = f.shape[0]    
            
    Kc    = covK.RBF(np.log(hyp[0]), np.log(hyp[1]))
    K     = Kc.getCovMatrix(x=x, mode='train')    
    n_nu = np.random.multivariate_normal(np.zeros_like(f), K, 1)
    nu = n_nu.T.reshape((n,))
    
    lp = -(y-f)**2 / sn**2/2 - np.log(2.*np.pi*sn**2)/2. - np.log(sn) - np.log(norm.cdf((4.6-f)/sn) - norm.cdf((0.-f)/sn))
    cur_logf = lp.sum()

    log_y = cur_logf + np.log(np.random.uniform())
    
    theta = np.random.uniform(high=2.*np.pi)
    theta_min = theta - 2. * np.pi
    theta_max = theta
    
    # Slice sampling loop
    while True:  
        prop_f = f * np.cos(theta) + nu * np.sin(theta)

        prop_lp = -(y-prop_f)**2 / sn**2/2 - np.log(2.*np.pi*sn**2)/2. - np.log(sn) - np.log(norm.cdf((4.6-prop_f)/sn) - norm.cdf((0.-prop_f)/sn))
        prop_logf = prop_lp.sum()

        if prop_logf > log_y and np.isfinite(prop_logf):                
            return prop_logf, prop_f

        else:
            if theta >= 0:
                theta_max = theta            
            else:
                theta_min = theta
            
            theta = np.random.uniform(low=theta_min, high=theta_max)


def surrogate_slice_sampling(var, sn, scale):
    '''
    Surrogate data slice sampling
    
    :param var[0] f: initial latent variable, shape (n,)
    :param var[1] x: input x, shape (n,k)
    :param var[2] y: input y, shape (n,)
    :param var[3] hyp: hyper-parameters to be updated, shape (2,)
    :param sn: noise in likelihood function
    :param sigma: scale, should be the same size of hyp

    :return prop_f: proposal latent variable, shape (n,)
    :return prop_hyp: proposal hyper-parameters
    '''
    f = var[0]
    x = var[1]
    y = var[2]
    hyp = var[3]

    Kc = covK.RBF(np.log(hyp[0]), np.log(hyp[1]))
    K  = Kc.getCovMatrix(x=x, mode='train') + np.eye(x.shape[0])*1e-10    # increase stability

    g, K_S, m_theta_g, chol_R_theta = aux_var_model(f, K, sn)
    ita = np.linalg.solve(chol_R_theta, f-m_theta_g)
    
    v = np.random.uniform(low=0., high=scale)
    hyp_min = np.maximum(hyp[0] - v, 0)
    hyp_max = hyp_min + scale

    llk = -(y-f)**2 / sn**2/2. - np.log(2.*np.pi*sn**2)/2. #- np.log(sn) - np.log(norm.cdf((4.6-f)/sn) - norm.cdf((0.-f)/sn))
    curLLK = llk.sum()

    curG = np.log(multivariate_normal.pdf(g, np.zeros_like(g), K_S))
    # curG = -(g.shape[0]*np.log(2*np.pi)/2. + np.log(np.diag(L_ks)).sum() + np.dot(np.dot(g.T, K_S), g)/2.)

    prior, junk = logGamma(hyp[0], 0.5, 2., False)
    threshold = np.log(np.random.uniform()) + curLLK + curG + prior

    while True:
        prop_hyp = np.random.uniform(low=hyp_min, high=hyp_max)
        # print 'min ', hyp_min
        # print prop_hyp
        # print 'max ', hyp_max

        Kp = covK.RBF(np.log(prop_hyp), np.log(hyp[1]))
        nK = Kp.getCovMatrix(x=x, mode='train') + np.eye(x.shape[0])*1e-10

        g, K_S, m_theta_g, chol_R_theta = aux_var_model(f, nK, sn, g=g)
        prop_f = np.dot(chol_R_theta, ita) + m_theta_g

        prop_llk = -(y-prop_f)**2 / sn**2/2 - np.log(2.*np.pi*sn**2)/2. #- np.log(sn) - np.log(norm.cdf((4.6-prop_f)/sn) - norm.cdf((0.-prop_f)/sn))
        propLLK = prop_llk.sum()

        propG = np.log(multivariate_normal.pdf(g, np.zeros_like(g), K_S))
        # propG = -(g.shape[0]*np.log(2*np.pi)/2. + np.log(np.diag(L_ks)).sum() + np.dot(np.dot(g.T, K_S), g)/2.)

        propPrior, junk = logGamma(prop_hyp, 0.5, 2., False)
        proposal = propLLK + propG + propPrior

        if proposal > threshold and np.isfinite(proposal):
            hyp[0] = prop_hyp
            return prop_f, hyp
        
        else:
            # for i in range(2):
            if prop_hyp < hyp[0]:
                hyp_min = prop_hyp
            else:
                hyp_max = prop_hyp


def aux_var_model(f, K, sn, g=None):
    '''
    auxiliary variable model p(f|g,theta) = N(f; m_theta_g, R_theta_g)

    :param f: original latent variables
    :param K: covariance of f
    :param alpha: auxiliary noise
    
    :return mean m_theta_g and covariance R_theta_g of auxiliary variable model
    '''
    n = K.shape[0]
    Kii = np.diagonal(K)
    K_ii_inv = 1./(Kii)
    v_1 = (sn**2)**(-1) + K_ii_inv        # v-1 = variance of posterior P(f|L, theta) according to Laplace Approximation
    Sii = 1./(v_1 - K_ii_inv)
    S = np.zeros_like(K)
    np.fill_diagonal(S, Sii)
    S = np.maximum(S, 0.)

    if g is None:
        # g = np.dot(tools.jitchol(K+S), np.random.normal(size=(n,)))
        g = np.random.multivariate_normal(f, S, 1).T.reshape((n,))

    # L = tools.jitchol(K+S)
    # V = np.linalg.solve(L, K)           # V = L-1 * K, V.T*V = K.T * (K+S)-1 * K
    # R_theta = K - np.dot(V.T, V)
    R_theta = K - np.dot(np.dot(K, np.linalg.inv(K+S)), K)

    # LS = np.linalg.cholesky(S)
    # beta = tools.solve_chol(LS.T, g)    # beta = S-1 * g
    # m_theta_g = np.dot(R_theta, beta)
    m_theta_g = np.dot(np.dot(R_theta, np.linalg.inv(S)), g)
    chol_R_theta = tools.jitchol(R_theta)
    
    return g, K+S, m_theta_g, chol_R_theta

    
def hmcK(x, E, var, leapfrog, epsilon, nsamples):
    '''
    try to implement hamiltonian monte carlo
    
    :param x: initial state
    :param E: function E(x) and its gradient
    :param var: parameters for computing log-likelihood and its gradient
    :param leapfrog: number of steps in leapfrog
    :param epsilon: step size in leapfrog
    :param nsamples: number of samples to be drawn
    
    :return new state, new llk
    '''
    n = x.shape[0]
    log, grad = E(x, var)      
    # E = -(llk + prior)
    log = 0. - log
    grad = 0. - grad
    assert log.shape == (n,) and grad.shape == (n,), 'Shape of llk or gradient doesn\'t match shape of input state'

    
    while nsamples > 0:
        p = np.random.randn(x.shape[0],)                     # initialize momentum with Normal(0,1)
        H = p*p / 2. + log                                   # compute the current hamiltonian function

        x_new = x
        grad_new = grad
        
        for tau in range(leapfrog):
            
            p = p - epsilon * grad_new / 2.                  # make half-step in p
            x_new = x_new + epsilon * p                      # make full step in x
            
            log_new, grad_new = E(x_new, var)                # find new gradient
            log_new = 0. - log_new
            grad_new = 0. - grad_new
            
            p = p - epsilon * grad_new / 2.

        H_new = p*p / 2. + log_new                           # compute new hamiltonian function
        delta_H = H_new - H                                  # decide whether to accept
        
        # length scale
        for i in range(n):
            if i == 2:
                print log[2], log_new[2]
            if delta_H[i] < 0 or np.random.rand() < np.exp(-delta_H[i]):
                grad[i] = grad_new[i]
                x[i] = x_new[i]
                log[i] = log_new[i]
                
                print 'accept! ', x
            else:
                print 'reject! ', x
        
        nsamples -= 1

    return x, log

            
def logp_hyper(state, var):
    '''
    log likelihood of hyperparameters and their gradients
    
    :param state: input state for HMC
    :param var: [latent variables (f), input x (x), input y (y), hyp (hyperparameter), option (opt)]
    :return: log probability, gradient
    '''
    f = var[0]
    x = var[1]
    y = var[2]
    hyp = state[:]
    
    
    # llk of GP distribution
    logN, gradN = logLikelihood(x, f, y, hyp)      # marginal llk
    
    # llk of gamma distribution
    k     = np.asarray([2., 2., 1.])
    theta = np.asarray([2., 2., 3.])
    logG, gradG = logGamma(state, k=k, theta=theta, invG=True)
    
    #print 'logN: ', logN, ' logG: ', logG
    logp = logN + logG
    grad = gradN + gradG
    
    return logp, grad


def logLikelihood(x, f, y, hyp):
    '''
    calculate log-likelihood of GP and truncated Gaussian distribution and its gradient
    i.e. log p( y | hyper )
    
    :param x: observation x
    :param f: latent variables
    :param y: observation y
    :param hyp: hyper in covariance
    :param K: covariance function of x
    :param option: which hyperparamter to be used for derivatives, must be ell, sf2, or noise
    
    :return: log-likelihood of normal distribution, its gradient w.r.t. hyperparameters
    '''
    sf2 = hyp[1]**2                                   # hyperparameter (sigma_y)^2 in RBF kernel (covariance function)
    sn2 = hyp[2]**2                                   # noise (sigma_n)^2
        
    covCur= covK.RBF(np.log(hyp[0]), np.log(hyp[1]))
    K     = covCur.getCovMatrix(x=x, mode='train')
    n     = np.shape(x)[0]
    L     = tools.jitchol(K/sn2+np.eye(n)).T          # K = L * L_T
    alpha = tools.solve_chol(L,y)/sn2                 # alpha = K**(-1) * f
    
    # log likelihood
    logN  = -(np.dot(y.T,alpha)/2. + np.log(np.diag(L)).sum() + n*np.log(2*np.pi*sn2)/2.)       # llk of hyper: p(y | theta)
    logN_n= np.sum(-(y-f)**2 / (2.*sn2) - 0.5*np.log(2.*np.pi*sn2))                             # llk of noise: p(y | f)
    
    logN  = np.asarray([logN, logN, logN_n])
        
    # gradient of llk
    Q = -(tools.solve_chol(L, np.eye(n))/sn2 - np.dot(alpha, alpha.T))    # precompute for convenience
    A = spdist.cdist(x/hyp[0],x/hyp[0],'sqeuclidean')
    derK = np.asarray([sf2 * np.exp(-0.5*A) * A * (1./hyp[0]),          # compute derivative matrix w.r.t. length scale
                       2. * hyp[1] * np.exp(-0.5*A),                    # compute derivative matrix w.r.t. signal y
                       ])
    assert derK[0].shape == (n,n) and derK[1].shape == (n,n), "Derivative of K has wrong shape!"

    gradN = np.zeros((3,))
    gradN[0] = (Q*derK[0]).sum()/2.
    gradN[1] = (Q*derK[1]).sum()/2.
#     gradN[2] = gradN_n.sum()
    gradN[2] = np.sum((y-f)**2 * hyp[2]**(-3) - hyp[2]**(-1))
    #gradN[2] = (np.trace(Q)*derK[2])/2.
        
    return logN, gradN                          


def logGamma(state, k, theta, invG):
    '''
    calculate log-likelihood of gamma distribution and the gradient of it
    
    :param x: hyperparameter to be updated
    :param k: shape parameter of gamma distribution
    :param theta: scale parameter of gamma distribution
    :param invG: whether to compute the prior of noise variance, i.e. inverse Gamma
    
    :return: log-likelihood of gamma distribution, its gradient w.r.t. hyperparameters
    '''
    # logG = np.log(gamma.pdf(x=state, a=k, loc=0, scale=theta))
    logG    = (k-1)*np.log(state) - state/theta - k*np.log(theta) - np.log(scipy.special.gamma(k))                              # Gamma prior
    gradG    = (k-1)*(1/state) - 1/theta

    if invG:
        logG[2] = np.log(theta[2]**k[2]) - np.log(scipy.special.gamma(k[2])) + (-k[2]-1)*np.log(state[2]) + (-theta[2]/state[2])    # Inverse-gamma prior
        gradG[2] = (-k[2]-1)/state[2] + theta[2]/(state[2]**2)

    return logG, gradG


def infMCMC(xs, sample_f, model):
    '''
    inference from fs|f
    '''
    x = model.x
    y = model.y
    n_samples = sample_f.shape[1]
    ns  = xs.shape[0]
    fmu = np.zeros((ns, 1))
    fs2 = np.zeros((ns, 1))

    n, D = x.shape
    m = model.meanfunc.getMean(x)
    K = model.covfunc.getCovMatrix(x=x, mode='train')
    sn2   = np.exp(2.*model.likfunc.hyp[0])
    L     = tools.jitchol(K/sn2+np.eye(n)).T
    alpha = tools.solve_chol(L, sample_f-m)/sn2
    sW    = np.ones((n, 1))/np.sqrt(sn2)

    Ltril = np.all(np.tril(L, -1) == 0)                         # is L an upper triangular matrix?
    kss = model.covfunc.getCovMatrix(z=xs, mode='self_test')
    Ks  = model.covfunc.getCovMatrix(x=x, z=xs, mode='cross')
    ms  = model.meanfunc.getMean(xs)
    Fmu = np.tile(ms, (1, n_samples)) + np.dot(Ks.T, alpha)     # conditional mean fs|f
    fmu = np.reshape(Fmu.sum(axis=1)/n_samples, (ns, 1))

    if Ltril: # L is triangular => use Cholesky parameters (alpha,sW,L)
        V   = np.linalg.solve(L.T,np.tile(sW,(1,ns))*Ks)
        fs2 = kss - np.array([(V*V).sum(axis=0)]).T
    else:     # L is not triangular => use alternative parametrization
        fs2 = kss + np.array([(Ks*np.dot(L,Ks)).sum(axis=0)]).T
    fs2 = np.maximum(fs2, 0)
    Fs2 = np.tile(fs2, (1, n_samples))

    trunclik = likK.TruncatedGauss(model.likfunc.hyp[0])
    junk, Ymu, Ys2 = trunclik.evaluate(None, Fmu[:], Fs2[:], None, None, 3)
    # Lp, Ymu, Ys2 = likfunc.evaluate(None,Fmu[:],Fs2[:],None,None,3)
    ym  = np.reshape(np.mean(Ymu, axis=1), (ns, 1))
    ys2 = np.reshape(np.mean(Ys2, axis=1), (ns, 1))

    print 'ns ', ns
    print 'Fmu ', Fmu.shape
    print 'Fs2 ', Fs2.shape
    print 'Ymu ', Ymu.shape
    print 'Ys2 ', Ys2.shape

    return ym, ys2, fmu, fs2


if __name__ == '__main__':
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
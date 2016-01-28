"""Implementation of Hybrid Monte Carlo sampling method, currently not in use"""
import numpy as np
import scipy.spatial.distance as spdist
from kcGP import covK, likK, tools

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


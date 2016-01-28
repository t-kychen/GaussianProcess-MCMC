"""
Created on Jul 5, 2015

Implementation of MCMC sampling:
    1) Elliptical slice sampling
    2) Surrogate data slice sampling

@author: K.Y. Chen
"""
import numpy as np
import scipy.spatial.distance as spdist
import scipy.special
from scipy.stats import norm
from kcGP import covK, likK, tools

def elliptical_slice(var, sn):
    """Implementation of elliptical slice sampling

    Reference: http://jmlr.csail.mit.edu/proceedings/papers/v9/murray10a/murray10a.pdf

    :param f: initial latent variable
    :param x: input x
    :param y: input y
    :param sn: parameter in likelihood func

    :return ll: log-likelihood
    :return ff: latent variable
    """
    f = var[0]
    x = var[1]
    y = var[2]
    my = np.mean(y)
    hyp = var[3]

    n = f.shape[0]    
            
    Kc    = covK.RBF(np.log(hyp[0]), np.log(hyp[1]))
    K     = Kc.getCovMatrix(x=x, mode='train')    
    n_nu = np.random.multivariate_normal(np.zeros_like(f), K, 1)
    nu = n_nu.T.reshape((n,))
    
    lp = -(y-my-f)**2 / sn**2/2 - np.log(2.*np.pi*sn**2)/2. - np.log(sn) - np.log(norm.cdf((4.6-f)/sn) - norm.cdf((0.-f)/sn))
    cur_logf = lp.sum()

    log_y = cur_logf + np.log(np.random.uniform())
    
    theta = np.random.uniform(high=2.*np.pi)
    theta_min = theta - 2. * np.pi
    theta_max = theta
    
    # Slice sampling loop
    while True:  
        prop_f = f * np.cos(theta) + nu * np.sin(theta)

        prop_lp = -(y-my-prop_f)**2 / sn**2/2 - np.log(2.*np.pi*sn**2)/2. - np.log(sn) - np.log(norm.cdf((4.6-prop_f)/sn) - norm.cdf((0.-prop_f)/sn))
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
    """Implementation of surrogate data slice sampling

    Reference: http://papers.nips.cc/paper/4114-slice-sampling-covariance-hyperparameters-of-latent-gaussian-models.pdf
    
    var[0] f: ndarray with shape (n_locations,)
        initial latent samples
    var[1] x: ndarray with shape (n_samples, n_locations)
        input samples
    var[2] y: ndarray with shape (n_locations,)
        input target values
    var[3] hyp: ndarray with shape (2,)
        hyper-parameters to be updated
    sn: float
        noise
    scale: ndarray with shape (n_hyperparams,)
        proposal ranges of hyper-parameters in SDS sampling update

    """
    f = var[0]
    x = var[1]
    y = var[2]
    my = np.mean(y)
    hyp = np.append(var[3], sn)

    Kc = covK.RBF(np.log(hyp[0]), np.log(hyp[1]))
    K  = Kc.getCovMatrix(x=x, mode='train')

    g, K_S, m_theta_g, chol_R_theta, L_ks = aux_var_model(f, K, hyp[2])
    ita = np.linalg.solve(chol_R_theta, f-m_theta_g)
    
    v = np.random.uniform(low=0., high=scale)
    hyp_min = np.maximum(hyp - v, 0)
    hyp_max = hyp_min + scale

    upper = 100. - my
    lower = 0. - my
    llk = -(y-my-f)**2 / hyp[2]**2/2. - np.log(2.*np.pi*hyp[2]**2)/2. - np.log(hyp[2]) - np.log(norm.cdf((upper-f)/hyp[2]) - norm.cdf((lower-f)/hyp[2]))
    curLLK = llk.sum()

    # curG = np.log(multivariate_normal.pdf(g, np.zeros_like(g), K_S))
    # alpha = tools.solve_chol(L_ks.T, g)
    # curG = -(np.dot(g.T, alpha)/2. + np.log(np.diag(L_ks.T)).sum() + g.shape[0]*np.log(2*np.pi)/2.)
    curG = -(np.dot(np.dot(g.T, np.linalg.inv(K_S)), g)/2. + np.log(np.diag(L_ks.T)).sum() + g.shape[0]*np.log(2*np.pi)/2.)

    k = np.asarray([50., 2., 4.])            # 9, 2, 3
    theta = np.asarray([0.1, 2., 0.001])     # 0.5, 2, 0.25
    prior, junk = logGamma(hyp, k, theta, True)
    threshold = np.log(np.random.uniform()) + curLLK + curG + prior[1] + prior[0] + prior[2]

    while True:
        prop_hyp = np.random.uniform(low=hyp_min, high=hyp_max)
        # prop_hyp[2] = sn

        Kp = covK.RBF(np.log(prop_hyp[0]), np.log(prop_hyp[1]))
        nK = Kp.getCovMatrix(x=x, mode='train')

        g, K_S, m_theta_g, chol_R_theta, L_ks = aux_var_model(f, nK, prop_hyp[2], g=g)
        prop_f = np.dot(chol_R_theta, ita) + m_theta_g

        prop_llk = -(y-my-prop_f)**2 / prop_hyp[2]**2/2 - np.log(2.*np.pi*prop_hyp[2]**2)/2. - np.log(prop_hyp[2]) - np.log(norm.cdf((upper-prop_f)/prop_hyp[2]) - norm.cdf((lower-prop_f)/prop_hyp[2]))
        propLLK = prop_llk.sum()

        # propG = np.log(multivariate_normal.pdf(g, np.zeros_like(g), K_S))
        # alpha = tools.solve_chol(L_ks.T, g)
        # propG = -(np.dot(g.T, alpha)/2. + np.log(np.diag(L_ks.T)).sum() + g.shape[0]*np.log(2*np.pi)/2.)
        propG = -(np.dot(np.dot(g.T, np.linalg.inv(K_S)), g)/2. + np.log(np.diag(L_ks.T)).sum() + g.shape[0]*np.log(2*np.pi)/2.)

        propPrior, junk = logGamma(prop_hyp, k, theta, True)
        proposal = propLLK + propG + propPrior[1] + propPrior[0] + propPrior[2]

        if proposal > threshold and np.isfinite(proposal):
            hyp = prop_hyp
            # hyp[2] = sn
            return prop_f, hyp
        
        else:
            for i in range(0, 3):
                if prop_hyp[i] < hyp[i]:
                    hyp_min[i] = prop_hyp[i]
                else:
                    hyp_max[i] = prop_hyp[i]

def aux_var_model(f, K, sn, g=None):
    """Implementation of auxiliary variable model inside sds sampling method

    Reference: http://papers.nips.cc/paper/4114-slice-sampling-covariance-hyperparameters-of-latent-gaussian-models.pdf

    P(f|g,theta) = N(f; m_theta_g, R_theta_g)

    :param f: original latent variables
    :param K: covariance of f
    :param alpha: auxiliary noise
    
    """
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

    L = tools.jitchol(K+S)
    V = np.linalg.solve(L, K)           # V = L-1 * K, V.T*V = K.T * (K+S)-1 * K
    R_theta = K - np.dot(V.T, V)
    # R_theta = K - np.dot(np.dot(K, np.linalg.inv(K+S)), K)

    # LS = np.linalg.cholesky(S)
    # beta = tools.solve_chol(LS.T, g)    # beta = S-1 * g
    # m_theta_g = np.dot(R_theta, beta)
    m_theta_g = np.dot(np.dot(R_theta, np.linalg.inv(S)), g)
    chol_R_theta = tools.jitchol(R_theta+np.eye(n)*1e-11)

    return g, K+S, m_theta_g, chol_R_theta, L

def logGamma(state, k, theta, invG):
    '''
    calculate log pdf of gamma distribution and the gradient of it
    
    :param state: hyperparameter to be updated
    :param k: shape parameter of gamma distribution
    :param theta: scale parameter of gamma distribution
    :param invG: whether to compute the prior of noise variance, i.e. inverse Gamma
    
    :return: log-likelihood of gamma distribution, its gradient w.r.t. hyperparameters
    '''
    logG  = (k-1)*np.log(state) - state/theta - k*np.log(theta) - np.log(scipy.special.gamma(k))                                    # Gamma prior
    gradG = (k-1)*(1/state) - 1/theta

    if invG:
        logG[2] = np.log(theta[2]**k[2]) - np.log(scipy.special.gamma(k[2])) + (-k[2]-1)*np.log(state[2]) + (-theta[2]/state[2])    # Inverse-gamma prior
        gradG[2] = (-k[2]-1)/state[2] + theta[2]/(state[2]**2)

    return logG, gradG

def infMCMC(xs, sample_f, model, ys=0):
    '''
    inference from fs|f
    '''
    x = model.x
    y = model.y
    my = np.mean(y)
    n_samples = sample_f.shape[1]
    ns  = xs.shape[0]

    n, D = x.shape
    m = np.tile(model.meanfunc.getMean(x), (1, n_samples))
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
        V   = np.linalg.solve(L.T,np.tile(sW, (1, ns))*Ks)
        fs2 = kss - np.array([(V*V).sum(axis=0)]).T
    else:     # L is not triangular => use alternative parametrization
        fs2 = kss + np.array([(Ks*np.dot(L, Ks)).sum(axis=0)]).T
    fs2 = np.maximum(fs2, 0)
    Fs2 = np.tile(fs2, (1, n_samples))

    trunclik = likK.TruncatedGauss(upper=100.-my, lower=0.-my,
                                   log_sigma=model.likfunc.hyp[0])
    lp, Ymu, Lower, Upper = trunclik.evaluateT(ys-my, Fmu[:], Fs2[:])
    ym  = np.reshape(np.mean(Ymu, axis=1), (ns, 1))
    ys_lw = np.reshape(np.mean(Lower, axis=1), (ns, 1))
    ys_up = np.reshape(np.mean(Upper, axis=1), (ns, 1))

    return ym, ys_lw, ys_up, fmu, fs2, lp

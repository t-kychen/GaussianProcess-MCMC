"""
    File name: sliceSample.py
    Author: Kuan-Yu Chen
    Python version: 2.7

    Implementation of MCMC sampling:
        1) Elliptical slice sampling: http://homepages.inf.ed.ac.uk/imurray2/pub/10ess/ess.pdf
        2) Surrogate data slice sampling: http://homepages.inf.ed.ac.uk/imurray2/pub/10hypers/hypers.pdf

"""
import numpy as np
import scipy.special
from kcGP import covK, likK, tools

def elliptical_slice(f, x, y, hyp):
    """Implementation of elliptical slice sampling

    Reference: http://jmlr.csail.mit.edu/proceedings/papers/v9/murray10a/murray10a.pdf

    Parameters
    ----------
    f: ndarray with shape (n_samples,)
        initial latent samples
    x: ndarray with shape (n_samples, n_features)
        input samples
    y: ndarray with shape (n_samples,)
        input target values
    hyp: ndarray with shape (3,)
        hyper-parameters to be updated

    Return
    ------

    """
    nobs = f.shape[0]
    my = np.mean(y)

    Kc   = covK.RBF(np.log(hyp[0]), np.log(hyp[1]))
    K    = Kc.getCovMatrix(x=x, mode='train')
    m    = np.zeros_like(f)
    n_nu = np.random.multivariate_normal(m, K, 1)

    nu = n_nu.T.reshape((nobs,))

    upper = 100 - my
    lower = 0 - my
    lik_func = likK.TruncatedGauss2(upper=upper, lower=lower, log_sigma=np.log(hyp[2]))
    # lik_func = likK.Gauss(log_sigma=np.log(hyp[2]))

    cur_llk = lik_func.evaluate(y=y-my, mu=f)
    cur_llk = cur_llk + np.log(np.random.uniform())
    # print 'current llk', cur_llk
    
    theta = np.random.uniform(high=2.*np.pi)
    theta_min = theta - 2. * np.pi
    theta_max = theta
    
    # Slice sampling loop
    while True:
        prop_f = f * np.cos(theta) + nu * np.sin(theta)

        prop_llk = lik_func.evaluate(y=y-my, mu=prop_f)

        if prop_llk > cur_llk and np.isfinite(prop_llk):
            # print 'proposed llk', prop_llk
            return prop_f

        else:
            if theta >= 0:
                theta_max = theta            
            else:
                theta_min = theta
            
            theta = np.random.uniform(low=theta_min, high=theta_max)

def surrogate_slice_sampling(f, x, y, hyp, scale, iter=0):
    """Implementation of surrogate data slice sampling

    Parameters
    ----------
    f: ndarray with shape (n_samples,)
        initial latent samples
    x: ndarray with shape (n_samples, n_features)
        training samples
    y: ndarray with shape (n_samples,)
        target values
    hyp: ndarray with shape (3,)
        hyper-parameters to be updated, including ll, sf, sn
    scale: ndarray with shape (n_hyperparameters,)
        proposal ranges of hyper-parameters in SDS sampling update
    iter: integer, default = 0
        current iteration of MCMC

    Return
    ----------
    prop_f: ndarray with shape (n_samples,)
        updated latent samples
    prop_hyp: ndarray with shape (3,)
        updated hyper-parameters, including ll, sf, sn

    """
    my = np.mean(y)

    Kc = covK.RBF(np.log(hyp[0]), np.log(hyp[1]))
    K  = Kc.getCovMatrix(x=x, mode='train')

    g, K_S, m_theta_g, chol_R_theta, L_ks = aux_var_model(f, K, hyp[2])
    ita = np.linalg.solve(chol_R_theta, f-m_theta_g)
    
    v = np.random.uniform(low=0., high=scale)
    hyp_min = np.maximum(hyp - v, 0)
    hyp_max = hyp_min + scale

    upper = 100 - my
    lower = 0 - my

    lik_func = likK.TruncatedGauss2(upper=upper, lower=lower, log_sigma=np.log(hyp[2]))
    cur_llk = lik_func.evaluate(y=y-my, mu=f)

    # alpha = tools.solve_chol(L_ks.T, g)
    # curG = -(np.dot(g.T, alpha)/2. + np.log(np.diag(L_ks.T)).sum() + g.shape[0]*np.log(2*np.pi)/2.)
    curG = -(np.dot(np.dot(g.T, np.linalg.inv(K_S)), g)/2. + np.log(np.diag(L_ks.T)).sum() + g.shape[0]*np.log(2*np.pi)/2.)

    k = np.asarray([1., 3., 3.])
    theta = np.asarray([1., 1.5, 3.])
    prior, junk = log_gamma(hyp, k, theta, True)
    threshold = np.log(np.random.uniform()) + cur_llk + prior[1] + prior[0] + curG
    if iter >= 500:
        threshold += prior[2]

    while True:
        prop_hyp = np.random.uniform(low=hyp_min, high=hyp_max)
        if iter < 500:
            prop_hyp[2] = hyp[2]

        Kp = covK.RBF(np.log(prop_hyp[0]), np.log(prop_hyp[1]))
        nK = Kp.getCovMatrix(x=x, mode='train')

        g, K_S, m_theta_g, chol_R_theta, L_ks = aux_var_model(f, nK, prop_hyp[2], g=g)
        prop_f = np.dot(chol_R_theta, ita) + m_theta_g

        lik_func.sn = prop_hyp[2]
        prop_llk = lik_func.evaluate(y=y-my, mu=prop_f)

        # alpha = tools.solve_chol(L_ks.T, g)
        # propG = -(np.dot(g.T, alpha)/2. + np.log(np.diag(L_ks.T)).sum() + g.shape[0]*np.log(2*np.pi)/2.)
        propG = -(np.dot(np.dot(g.T, np.linalg.inv(K_S)), g)/2. + np.log(np.diag(L_ks.T)).sum() + g.shape[0]*np.log(2*np.pi)/2.)

        propPrior, junk = log_gamma(prop_hyp, k, theta, True)
        proposal = prop_llk + propPrior[1] + propPrior[0] + propG
        if iter >= 500:
            proposal += propPrior[2]

        if proposal > threshold and np.isfinite(proposal):

            return prop_f, prop_hyp

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

    Parameters
    ----------
    f: ndarray with shape (n_samples,)
        latent samples
    K: ndarray with shape (n_samples, n_samples)
        covariance of training samples
    sn: float
        noise
    g: ndarray with shape (n_samples,)
        auxiliary values with P(g|theta) = N(g; 0, Sigma_theta + S_theta)
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

def log_gamma(x, k, theta, invG):
    """Log PDF of Gamma and Inverse Gamma distributions and gradients of them

    Parameters
    ----------
    x: ndarray with shape (n_hyperparameters,)
        hyper-parameter to be updated
    k: ndarray with shape (n_hyperparameters,)
        shape parameters of gamma & inverse gamma
    theta: ndarray with shape (n_hyperparameters,)
        scale parameter of gamma distribution
    invG: boolean
        whether to compute the inverse gamma prior
    """
    # Gamma prior
    logG  = (k-1)*np.log(x) - x/theta - k*np.log(theta) - np.log(scipy.special.gamma(k))
    gradG = (k-1)*(1/x) - 1/theta

    # Inverse-gamma prior
    if invG:
        logG[2] = np.log(theta[2]**k[2]) - np.log(scipy.special.gamma(k[2])) + (-k[2]-1)*np.log(x[2]) + (-theta[2]/x[2])
        gradG[2] = (-k[2]-1)/x[2] + theta[2]/(x[2]**2)

    return logG, gradG

def inf_mcmc(f, model, ys=0):
    """Inference of fs|f

    Parameters
    ----------
    f: ndarray with shape (n_samples, n_mcmc_iters)
        latent samples from MCMC
    model: GP instance
        Trained Gaussian process model, which has x, y, mean, cov & llk func's
    ys: ndarray with shape (n_samples,)
        testing target, for the purpose of computing log-likelihood
    """
    x = model.x
    y = model.y
    xs= model.xs
    my = np.mean(y)
    n_samples = f.shape[1]
    ns  = xs.shape[0]

    n, D = x.shape
    m = np.tile(model.meanfunc.getMean(x), (1, n_samples))
    K = model.covfunc.getCovMatrix(x=x, mode='train')
    sn2   = model.likfunc.sn**2.
    L     = tools.jitchol(K/sn2+np.eye(n)).T
    alpha = tools.solve_chol(L, f-m)/sn2            # np.dot(Sigma**(-1), f-m)
    sW    = np.ones((n, 1))/np.sqrt(sn2)

    Ltril = np.all(np.tril(L, -1) == 0)                         # is L an upper triangular matrix?
    kss = model.covfunc.getCovMatrix(z=xs, mode='self_test')    # this only contains the diagonal terms
    Ks  = model.covfunc.getCovMatrix(x=x, z=xs, mode='cross')

    ms  = model.meanfunc.getMean(xs)
    Fmu = np.tile(ms, (1, n_samples)) + np.dot(Ks.T, alpha)     # conditional mean fs|f

    if Ltril: # L is triangular => use Cholesky parameters (alpha, sW, L)
        V   = np.linalg.solve(L.T, np.tile(sW, (1, ns))*Ks)                     # Ks has shape (n, ns),
        fs2 = kss - np.array([(V*V).sum(axis=0)]).T
    else:     # L is not triangular => use alternative parametrization
        fs2 = kss + np.array([(Ks*np.dot(L, Ks)).sum(axis=0)]).T

    # variance can only be >= 0
    Fs2 = np.maximum(fs2, 0)
    # Fs2 = np.tile(fs2, (1, n_samples))
    Fmu = np.mean(Fmu, axis=1, keepdims=True)

    Ymu, Lower, Upper = model.likfunc.evaluate(mu=Fmu, s2=Fs2)
    ym  = np.reshape(np.mean(Ymu, axis=1), (ns, 1)) + my
    ys_lw = np.reshape(np.mean(Lower, axis=1), (ns, 1)) + my
    ys_up = np.reshape(np.mean(Upper, axis=1), (ns, 1)) + my

    return ym, ys_lw, ys_up, Fs2

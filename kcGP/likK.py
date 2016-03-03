'''
Created on Jun 30, 2015
@author: Thomas
@Copyright by Marion Neumann and Shan Huang, 30/09/2013
http://www-ai.cs.uni-dortmund.de/weblab/static/api_docs/pyGPs/
'''
import infK
import numpy as np
from scipy.special import erf
from scipy.stats import norm

class Likelihood(object):
    '''
    Base function for Likelihood function
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.hyp = []

    def evaluate(self, y=None, mu=None, s2=None, inffunc=None, der=None, nargout=1):
        '''
        The likelihood functions have two possible modes, the mode being selected
        as follows:
        1) With two or three input arguments:                       [PREDICTION MODE]
         lp = evaluate(y, mu) OR lp, ymu, ys2 = evaluate(y, mu, s2)
            This allows to evaluate the predictive distribution. Let p(y_*|f_*) be the
            likelihood of a test point and N(f_*|mu,s2) an approximation to the posterior
            marginal p(f_*|x_*,x,y) as returned by an inference method. The predictive
            distribution p(y_*|x_*,x,y) is approximated by:
            q(y_*) = \int N(f_*|mu,s2) p(y_*|f_*) df_*
            lp = log( q(y) ) for a particular value of y, if s2 is [] or 0, this
            corresponds to log( p(y|mu) ).
            ymu and ys2 are the mean and variance of the predictive marginal q(y)
            note that these two numbers do not depend on a particular
            value of y.
            All vectors have the same size.
        2) With four or five input arguments, the fouth being an object of class "Inference" [INFERENCE MODE]
         evaluate(y, mu, s2, inf.EP()) OR evaluate(y, mu, s2, inf.Laplace(), i)
         There are two cases for inf, namely a) infLaplace, b) infEP
         The last input i, refers to derivatives w.r.t. the ith hyperparameter.
         | a1)
         | lp,dlp,d2lp,d3lp = evaluate(y, f, [], inf.Laplace()).
         | lp, dlp, d2lp and d3lp correspond to derivatives of the log likelihood.
         | log(p(y|f)) w.r.t. to the latent location f.
         | lp = log( p(y|f) )
         | dlp = d log( p(y|f) ) / df
         | d2lp = d^2 log( p(y|f) ) / df^2
         | d3lp = d^3 log( p(y|f) ) / df^3
         | a2)
         | lp_dhyp,dlp_dhyp,d2lp_dhyp = evaluate(y, f, [], inf.Laplace(), i)
         | returns derivatives w.r.t. to the ith hyperparameter
         | lp_dhyp = d log( p(y|f) ) / (dhyp_i)
         | dlp_dhyp = d^2 log( p(y|f) ) / (df   dhyp_i)
         | d2lp_dhyp = d^3 log( p(y|f) ) / (df^2 dhyp_i)
         | b1)
         | lZ,dlZ,d2lZ = evaluate(y, mu, s2, inf.EP())
         | let Z = \int p(y|f) N(f|mu,s2) df then
         | lZ = log(Z)
         | dlZ = d log(Z) / dmu
         | d2lZ = d^2 log(Z) / dmu^2
         | b2)
         | dlZhyp = evaluate(y, mu, s2, inf.EP(), i)
         | returns derivatives w.r.t. to the ith hyperparameter
         | dlZhyp = d log(Z) / dhyp_i
        Cumulative likelihoods are designed for binary classification. Therefore, they
        only look at the sign of the targets y; zero values are treated as +1.
        Some examples for valid likelihood functions:
         | lik = Gauss([0.1])
         | lik = Erf()
        '''
        pass

class Gauss(Likelihood):
    '''
    Gaussian likelihood function for regression.
    :math:`Gauss(t)=\\frac{1}{\\sqrt{2\\pi\\sigma^2}}e^{-\\frac{(t-y)^2}{2\\sigma^2}}`,
    where :math:`y` is the mean and :math:`\\sigma` is the standard deviation.
    hyp = [ log_sigma ]
    '''

    def __init__(self, log_sigma=np.log(0.1)):
        self.hyp = [log_sigma]

    def evaluate(self, y=None, mu=None, s2=None, inffunc=None, der=None, nargout=1):
        sn2 = np.exp(2. * self.hyp[0])

        if inffunc == None:             # Prediction mode
            if y is None:
                y = np.zeros_like(mu)

            s2zero = True
            if not (s2 is None) and np.linalg.norm(s2) > 0:
                s2zero = False

            if s2zero:
                lp = -(y-mu)**2 / sn2/2 - np.log(2.*np.pi*sn2)/2.   # p = 1/(sqrt(2*pi*sn2)) * exp(-(y-mu)**2/sn2/2)
                s2 = np.zeros_like(s2)
            else:
                inf_func = infK.EP()
                lp = self.evaluate(y, mu, s2, inf_func)

            if nargout > 1:
                ymu = mu
                if nargout > 2:
                    ys2 = s2 + sn2
                    return lp, ymu, ys2
                else:
                    return lp, ymu
            else:
                return lp

        else:
            if isinstance(inffunc, infK.EP):
                if der is None:                                  # no derivative mode
                    lZ = -(y-mu)**2/(sn2+s2)/2. - np.log(2*np.pi*(sn2+s2))/2. # log part function
                    if nargout>1:
                        dlZ  = (y-mu)/(sn2+s2)                   # 1st derivative w.r.t. mean
                        if nargout>2:
                            d2lZ = -1/(sn2+s2)                   # 2nd derivative w.r.t. mean
                            return lZ,dlZ,d2lZ
                        else:
                            return lZ,dlZ
                    else:
                        return lZ
                else:                                            # derivative mode
                    dlZhyp = ((y-mu)**2/(sn2+s2)-1) / (1+s2/sn2) # deriv. w.r.t. hyp.lik
                    return dlZhyp

            elif isinstance(inffunc, infK.Laplace):
                if der is None:                                  # no derivative mode
                    if y is None:
                        y=0
                    ymmu = y-mu
                    lp = -ymmu**2/(2*sn2) - np.log(2*np.pi*sn2)/2.
                    if nargout>1:
                        dlp = ymmu/sn2                           # dlp, derivative of log likelihood
                        if nargout>2:                            # d2lp, 2nd derivative of log likelihood
                            d2lp = -np.ones_like(ymmu)/sn2
                            if nargout>3:                        # d3lp, 3rd derivative of log likelihood
                                d3lp = np.zeros_like(ymmu)
                                return lp,dlp,d2lp,d3lp
                            else:
                                return lp,dlp,d2lp
                        else:
                            return lp,dlp
                    else:
                        return lp
                else:                                            # derivative mode
                    lp_dhyp   = (y-mu)**2/sn2 - 1                # derivative of log likelihood w.r.t. hypers
                    dlp_dhyp  = 2*(mu-y)/sn2                     # first derivative,
                    d2lp_dhyp = 2*np.ones_like(mu)/sn2           # and also of the second mu derivative
                    return lp_dhyp,dlp_dhyp,d2lp_dhyp

class TruncatedGauss(Likelihood):
    '''
    Truncated Gaussian likelihood function.
    '''
    def __init__(self, upper, lower, log_sigma=np.log(0.1)):
        '''
        :param upper: float, upper limit of truncated Gaussian
        :param lower: float, lower limit of truncated Gaussian
        :param log_sigma: float, logarithmic noise of Gaussian distribution
        '''
        self.upper = upper
        self.lower = lower
        self.sn = np.exp(log_sigma)

    def evaluate(self, y=None, mu=None, s2=None, inffunc=None, der=None, nargout=1):
        '''
        This function serves two purposes:
            1) If y is given, calculate log-likelihood.
            2) Given fs (including m(x_s) & k(x_s, x_s'), return ys | fs ~ N(0, np.exp(2.*log_sigma)) with truncation at 0 & 100
        :param y:
        :param mu:
        :param s2:
        :return:
        '''
        conf_lv = 0.05
        noise = self.sn**2.
        if len(mu.shape) == 1:
            mu = mu.reshape((mu.shape[0], 1))

        mu = np.mean(mu, axis=1)

        # log likelihood
        if not (y is None):
            llk = -(y-mu)**2 / noise/2. - np.log(2.*np.pi*noise)/2. - np.log(noise**0.5) - \
                  np.log(norm.cdf((self.upper-mu)/noise**0.5) - norm.cdf((self.lower-mu)/noise**0.5))
            llk = np.sum(llk)

            return llk

        beta  = (self.upper-mu)/noise**0.5
        alpha = (self.lower-mu)/noise**0.5
        Z     = norm.cdf(beta) - norm.cdf(alpha)

        ymu = mu + noise**0.5*(norm.pdf(alpha)-norm.pdf(beta))/Z
        cdf_lw = conf_lv*Z/2. + norm.cdf(alpha)
        ys2_lower = norm.ppf(cdf_lw) * noise**0.5 + mu

        cdf_up = (1-conf_lv/2.)*Z + norm.cdf(alpha)
        ys2_upper = norm.ppf(cdf_up) * noise**0.5 + mu

        return ymu, ys2_lower, ys2_upper

class TruncatedGaussNew(Likelihood):
    '''
    Truncated Gaussian likelihood function.
    '''
    def __init__(self, upper, lower, log_sigma=np.log(0.1)):
        '''
        :param upper: float, upper limit of truncated Gaussian
        :param lower: float, lower limit of truncated Gaussian
        :param log_sigma: float, logarithmic noise of Gaussian distribution
        '''
        self.upper = upper
        self.lower = lower
        self.sn = np.exp(log_sigma)

    def evaluate(self, y=None, mu=None, s2=None, inffunc=None, der=None, nargout=1):
        '''
        This function serves two purposes:
            1) If y is given, calculate log-likelihood.
            2) Given fs (including m(x_s) & k(x_s, x_s'), return ys | fs ~ N(0, np.exp(2.*log_sigma)) with truncation at 0 & 100
        :param y:
        :param mu:
        :param s2:
        :return:
        '''
        # log likelihood
        noise = self.sn**2.
        if not (y is None):
            if len(mu.shape) == 1:
                mu = mu.reshape((mu.shape[0], 1))

            mu = np.mean(mu, axis=1)

            llk = -(y-mu)**2 / noise/2. - np.log(2.*np.pi*noise)/2. - np.log(noise**0.5)

            idx_y_upper = (y>=self.upper)
            idx_y_lower = (y<=self.lower)

            llk[idx_y_upper] = 1 - norm.cdf((self.upper-mu[idx_y_upper])/noise**0.5)
            llk[idx_y_lower] = norm.cdf((self.lower-mu[idx_y_lower])/noise**0.5)

            llk = np.sum(llk)
            return llk

        ymu = np.minimum(np.maximum(mu, self.lower), self.upper)
        ys2 = s2 + noise
        ys2_lower = np.maximum(ymu-ys2, self.lower)
        ys2_upper = np.minimum(ymu+ys2, self.upper)

        return ymu, ys2_lower, ys2_upper

class Erf(Likelihood):
    '''
    Error function or cumulative Gaussian likelihood function for binary
    classification or probit regression.
    :math:`Erf(t)=\\frac{1}{2}(1+erf(\\frac{t}{\\sqrt{2}}))=normcdf(t)`
    '''
    def __init__(self):
        self.hyp = []

    def evaluate(self, y=None, mu=None, s2=None, inffunc=None, der=None, nargout=1):
        if not y is None:
            y = np.sign(y)
            y[y==0] = 1
        else:
            y = 1                                        # allow only +/- 1 values
        if inffunc is None:                              # prediction mode if inf is not present
            y = y*np.ones_like(mu)                       # make y a vector
            s2zero = True;
            if not s2 is None:
                if np.linalg.norm(s2)>0:
                    s2zero = False                       # s2==0?
            if s2zero:                                   # log probability evaluation
                p,lp = self.cumGauss(y,mu,2)
            else:                                        # prediction
                lp = self.evaluate(y, mu, s2, infK.EP())
                p = np.exp(lp)
            if nargout>1:
                ymu = 2*p-1                              # first y moment
                if nargout>2:
                    ys2 = 4*p*(1-p)                      # second y moment
                    return lp,ymu,ys2
                else:
                    return lp,ymu
            else:
                return lp
        else:                                            # inference mode
            if isinstance(inffunc, infK.Laplace):
                if der is None:                          # no derivative mode
                    f = mu; yf = y*f                     # product latents and labels
                    p,lp = self.cumGauss(y,f,2)
                    if nargout>1:                        # derivative of log likelihood
                        n_p = self.gauOverCumGauss(yf,p)
                        dlp = y*n_p                      # derivative of log likelihood
                        if nargout>2:                    # 2nd derivative of log likelihood
                            d2lp = -n_p**2 - yf*n_p
                            if nargout>3:                # 3rd derivative of log likelihood
                                d3lp = 2*y*n_p**3 + 3*f*n_p**2 + y*(f**2-1)*n_p
                                return lp,dlp,d2lp,d3lp
                            else:
                                return lp,dlp,d2lp
                        else:
                            return lp,dlp
                    else:
                        return lp
                else:                                    # derivative mode
                    return []                            # derivative w.r.t. hypers

            elif isinstance(inffunc, infK.EP):
                if der is None:                          # no derivative mode
                    z = mu/np.sqrt(1+s2)
                    junk, lZ = self.cumGauss(y,z,2)       # log part function
                    if not y is None:
                        z = z*y
                    if nargout>1:
                        if y is None: y = 1
                        n_p = self.gauOverCumGauss(z,np.exp(lZ))
                        dlZ = y*n_p/np.sqrt(1.+s2)       # 1st derivative wrt mean
                        if nargout>2:
                            d2lZ = -n_p*(z+n_p)/(1.+s2)  # 2nd derivative wrt mean
                            return lZ,dlZ,d2lZ
                        else:
                            return lZ,dlZ
                    else:
                        return lZ
                else:                                    # derivative mode
                    return []                       # deriv. wrt hyp.lik


    def cumGauss(self, y=None, f=None, nargout=1):
        # return [p,lp] = cumGauss(y,f)
        if not y is None:
            yf = y*f
        else:
            yf = f
        p = (1. + erf(yf/np.sqrt(2.)))/2. # likelihood
        if nargout>1:
            lp = self.logphi(yf,p)
            return p,lp
        else:
            return p

    def gauOverCumGauss(self,f,p):
        # return n_p = gauOverCumGauss(f,p)
        n_p = np.zeros_like(f)       # safely compute Gaussian over cumulative Gaussian
        ok = f>-5                    # naive evaluation for large values of f
        n_p[ok] = (np.exp(-f[ok]**2/2)/np.sqrt(2*np.pi)) / p[ok]
        bd = f<-6                    # tight upper bound evaluation
        n_p[bd] = np.sqrt(f[bd]**2/4+1)-f[bd]/2
        interp = np.logical_and(np.logical_not(ok),np.logical_not(bd)) # linearly interpolate between both of them
        tmp = f[interp]
        lam = -5. - f[interp]
        n_p[interp] = (1-lam)*(np.exp(-tmp**2/2)/np.sqrt(2*np.pi))/p[interp] + lam *(np.sqrt(tmp**2/4+1)-tmp/2);
        return n_p

    def logphi(self,z,p):
        # return lp = logphi(z,p)
        lp = np.zeros_like(z)                       # allocate memory
        zmin = -6.2; zmax = -5.5;
        ok = z>zmax                                 # safe evaluation for large values
        bd = z<zmin                                 # use asymptotics
        nok = np.logical_not(ok)
        ip = np.logical_and(nok,np.logical_not(bd)) # interpolate between both of them
        lam = 1/(1.+np.exp( 25.*(0.5-(z[ip]-zmin)/(zmax-zmin)) ))  # interp. weights
        lp[ok] = np.log(p[ok])
        lp[nok] = -np.log(np.pi)/2. -z[nok]**2/2. - np.log( np.sqrt(z[nok]**2/2.+2.) - z[nok]/np.sqrt(2.) )
        lp[ip] = (1-lam)*lp[ip] + lam*np.log( p[ip] )
        return lp

class Laplace(Likelihood):
    '''
    Laplacian likelihood function for regression. ONLY works with EP inference!
    :math:`Laplace(t) = \\frac{1}{2b}e^{-\\frac{|t-y|}{b}}` where :math:`b=\\frac{\\sigma}{\\sqrt{2}}`,
    :math:`y` is the mean and :math:`\\sigma` is the standard deviation.
    hyp = [ log_sigma ]
    '''
    def __init__(self, log_sigma=np.log(0.1) ):
        self.hyp = [ log_sigma ]

    def evaluate(self, y=None, mu=None, s2=None, inffunc=None, der=None, nargout=1):
        sn = np.exp(self.hyp); b = sn/np.sqrt(2);
        if y is None:
            y = np.zeros_like(mu)
        if inffunc is None:                              # prediction mode if inf is not present
            if y is None:
                y = np.zeros_like(mu)
            s2zero = True;
            if not s2 is None:
                if np.linalg.norm(s2)>0:
                    s2zero = False                       # s2==0?
            if s2zero:                                   # log probability evaluation
                lp = -np.abs(y-mu)/b -np.log(2*b); s2 = 0
            else:                                        # prediction
                lp = self.evaluate(y, mu, s2, infK.EP())
            if nargout>1:
                ymu = mu                              # first y moment
                if nargout>2:
                    ys2 = s2 + sn**2                  # second y moment
                    return lp,ymu,ys2
                else:
                    return lp,ymu
            else:
                return lp
        else:                                            # inference mode
            if isinstance(inffunc, infK.Laplace):
                if der is None:                          # no derivative mode
                    if y is None:
                        y = np.zeros_like(mu)
                    ymmu = y-mu
                    lp = np.abs(ymmu)/b - np.log(2*b)
                    if nargout>1:                        # derivative of log likelihood
                        dlp = np.sign(ymmu)/b
                        if nargout>2:                    # 2nd derivative of log likelihood
                            d2lp = np.zeros_like(ymmu)
                            if nargout>3:                # 3rd derivative of log likelihood
                                d3lp = np.zeros_like(ymmu)
                                return lp,dlp,d2lp,d3lp
                            else:
                                return lp,dlp,d2lp
                        else:
                            return lp,dlp
                    else:
                        return lp
                else:                                    # derivative w.r.t. hypers
                    lp_dhyp = np.abs(y-mu)/b - 1           # derivative of log likelihood w.r.t. hypers
                    dlp_dhyp = np.sign(mu-y)/b              # first derivative,
                    d2lp_dhyp = np.zeros(mu.shape)         # and also of the second mu derivative
                    return lp_dhyp, dlp_dhyp, d2lp_dhyp
            elif isinstance(inffunc, infK.EP):
                n = np.max([len(y.flatten()),len(mu.flatten()),len(s2.flatten()),len(sn.flatten())])
                on = np.ones((n,1))
                y = y*on; mu = mu*on; s2 = s2*on; sn = sn*on;
                fac = 1e3;          # factor between the widths of the two distributions ...
                                    # ... from when one considered a delta peak, we use 3 orders of magnitude
                #idlik = np.reshape( (fac*sn) < np.sqrt(s2) , (sn.shape[0],) ) # Likelihood is a delta peak
                #idgau = np.reshape( (fac*np.sqrt(s2)) < sn , (sn.shape[0],) ) # Gaussian is a delta peak
                idlik = (fac*sn) < np.sqrt(s2)
                idgau = (fac*np.sqrt(s2)) < sn
                id    = np.logical_and(np.logical_not(idgau),np.logical_not(idlik)) # interesting case in between

                if der is None:                          # no derivative mode
                    lZ = np.zeros((n,1))
                    dlZ = np.zeros((n,1))
                    d2lZ = np.zeros((n,1))
                    if np.any(idlik):
                        l = Gauss(log_sigma=np.log(s2[idlik])/2)
                        a = l.evaluate(mu[idlik], y[idlik])
                        lZ[idlik] = a[0]; dlZ[idlik] = a[1]; d2lZ[idlik] = a[2]
                    if np.any(idgau):
                        l = Laplace(log_hyp=np.log(sn[idgau]))
                        a = l.evaluate(mu=mu[idgau], y=y[idgau])
                        lZ[idgau] = a[0]; dlZ[idgau] = a[1]; d2lZ[idgau] = a[2]
                    if np.any(id):
                        # substitution to obtain unit variance, zero mean Laplacian
                        tvar = s2[id]/(sn[id]**2+1e-16)
                        tmu = (mu[id]-y[id])/(sn[id]+1e-16)
                        # an implementation based on logphi(t) = log(normcdf(t))
                        zp = (tmu+np.sqrt(2)*tvar)/np.sqrt(tvar)
                        zm = (tmu-np.sqrt(2)*tvar)/np.sqrt(tvar)
                        ap =  self._logphi(-zp)+np.sqrt(2)*tmu
                        am =  self._logphi( zm)-np.sqrt(2)*tmu
                        apam = np.vstack((ap,am)).T
                        lZ[id] = self._logsum2exp(apam) + tvar - np.log(sn[id]*np.sqrt(2.))

                    if nargout>1:
                        lqp = -0.5*zp**2 - 0.5*np.log(2*np.pi) - self._logphi(-zp);       # log( N(z)/Phi(z) )
                        lqm = -0.5*zm**2 - 0.5*np.log(2*np.pi) - self._logphi( zm);
                        dap = -np.exp(lqp-0.5*np.log(s2[id])) + np.sqrt(2)/sn[id]
                        dam =  np.exp(lqm-0.5*np.log(s2[id])) - np.sqrt(2)/sn[id]
                        _z1 = np.vstack((ap,am)).T
                        _z2 = np.vstack((dap,dam)).T
                        _x = np.array([[1],[1]])
                        dlZ[id] = self._expABz_expAx(_z1, _x, _z2, _x)
                        if nargout>2:
                            a = np.sqrt(8.)/sn[id]/np.sqrt(s2[id]);
                            bp = 2./sn[id]**2 - (a - zp/s2[id])*np.exp(lqp)
                            bm = 2./sn[id]**2 - (a + zm/s2[id])*np.exp(lqm)
                            _x = np.reshape(np.array([1,1]),(2,1))
                            _z1 = np.reshape(np.array([ap,am]),(1,2))
                            _z2 = np.reshape(np.array([bp,bm]),(1,2))
                            d2lZ[id] = self._expABz_expAx(_z1, _x, _z2, _x) - dlZ[id]**2
                            return lZ,dlZ,d2lZ
                        else:
                            return lZ,dlZ
                    else:
                        return lZ
                else:                                    # derivative mode
                    dlZhyp = np.zeros((n,1))
                    if np.any(idlik):
                        dlZhyp[idlik] = 0
                    if np.any(idgau):
                        l = Laplace(log_hyp=np.log(sn[idgau]))
                        a =  l.evaluate(mu=mu[idgau], y=y[idgau], inffunc='inf.Laplace', nargout=1)
                        dlZhyp[idgau] = a[0]

                    if np.any(id):
                        # substitution to obtain unit variance, zero mean Laplacian
                        tmu = (mu[id]-y[id])/(sn[id]+1e-16);        tvar = s2[id]/(sn[id]**2+1e-16)
                        zp  = (tvar+tmu/np.sqrt(2))/np.sqrt(tvar);  vp = tvar+np.sqrt(2)*tmu
                        zm  = (tvar-tmu/np.sqrt(2))/np.sqrt(tvar);  vm = tvar-np.sqrt(2)*tmu
                        dzp = (-s2[id]/sn[id]+tmu*sn[id]/np.sqrt(2)) / np.sqrt(s2[id])
                        dvp = -2*tvar - np.sqrt(2)*tmu
                        dzm = (-s2[id]/sn[id]-tmu*sn[id]/np.sqrt(2)) / np.sqrt(s2[id])
                        dvm = -2*tvar + np.sqrt(2)*tmu
                        lezp = self._lerfc(zp); # ap = exp(vp).*ezp
                        lezm = self._lerfc(zm); # am = exp(vm).*ezm
                        vmax = np.max(np.array([vp+lezp,vm+lezm]),axis=0); # subtract max to avoid numerical pb
                        ep  = np.exp(vp+lezp-vmax)
                        em  = np.exp(vm+lezm-vmax)
                        dap = ep*(dvp - 2/np.sqrt(np.pi)*np.exp(-zp**2-lezp)*dzp)
                        dam = em*(dvm - 2/np.sqrt(np.pi)*np.exp(-zm**2-lezm)*dzm)
                        dlZhyp[id] = (dap+dam)/(ep+em) - 1;
                    return dlZhyp               # deriv. wrt hyp.lik
            elif isinstance(inffunc, infK.VB):
                n = len(s2.flatten()); b = np.zeros((n,1)); y = y*np.ones((n,1)); z = y
                return b,z

    def _lerfc(self,t):
        ''' numerically safe implementation of f(t) = log(1-erf(t)) = log(erfc(t))'''
        from scipy.special import erfc
        f  = np.zeros_like(t)
        tmin = 20; tmax = 25
        ok = t<tmin                              # log(1-erf(t)) is safe to evaluate
        bd = t>tmax                              # evaluate tight bound
        nok = np.logical_not(ok)
        interp = np.logical_and(nok,np.logical_not(bd)) # interpolate between both of them
        f[nok] = np.log(2/np.sqrt(np.pi)) -t[nok]**2 -np.log(t[nok]+np.sqrt( t[nok]**2+4/np.pi ))
        lam = 1/(1+np.exp( 12*(0.5-(t[interp]-tmin)/(tmax-tmin)) ))   # interp. weights
        f[interp] = lam*f[interp] + (1-lam)*np.log(erfc( t[interp] ))
        f[ok] += np.log(erfc( t[ok] ))             # safe eval
        return f

    def _expABz_expAx(self,A,x,B,z):
        '''
        Computes y = ( (exp(A).*B)*z ) ./ ( exp(A)*x ) in a numerically safe way
        The function is not general in the sense that it yields correct values for
        all types of inputs. We assume that the values are close together.
        '''
        N = A.shape[1]
        maxA = np.max(A,axis=1)                    # number of columns, max over columns
        maxA = np.array([maxA]).T
        A = A - np.dot(maxA, np.ones((1,N)))       # subtract maximum value
        y = ( np.dot((np.exp(A)*B),z) ) / ( np.dot(np.exp(A),x) )
        return y[0]

    def _logphi(self,z):
        ''' Safe implementation of the log of phi(x) = \int_{-\infty}^x N(f|0,1) df
         returns lp = log(normcdf(z))
        '''
        lp = np.zeros_like(z)                       # allocate memory
        zmin = -6.2; zmax = -5.5;
        ok = z>zmax                                 # safe evaluation for large values
        bd = z<zmin                                 # use asymptotics
        nok = np.logical_not(ok)
        ip = np.logical_and(nok,np.logical_not(bd)) # interpolate between both of them
        lam = 1./(1.+np.exp( 25.*(0.5-(z[ip]-zmin)/(zmax-zmin)) ))  # interp. weights
        lp[ok] = np.log( 0.5*( 1.+erf(z[ok]/np.sqrt(2.)) ) )
        lp[nok] = -0.5*(np.log(np.pi) + z[nok]**2) - np.log( np.sqrt(2.+0.5*(z[nok]**2)) - z[nok]/np.sqrt(2))
        lp[ip] = (1-lam)*lp[ip] + lam*np.log( 0.5*( 1.+erf(z[ip]/np.sqrt(2.)) ) )
        return lp

    def _logsum2exp(self,logx):
        '''computes y = log( sum(exp(x),2) ) in a numerically safe way
        by subtracting the row maximum to avoid cancelation after taking
        the exp the sum is done along the rows'''
        N = logx.shape[1]
        max_logx = logx.max(1)
        max_logx = np.array([max_logx]).T
        # we have all values in the log domain, and want to calculate a sum
        x = np.exp(logx - np.dot(max_logx,np.ones((1,N))))
        y = np.log(np.array([np.sum(x,1)]).T) + max_logx
        return list(y.flatten())

if __name__ == '__main__':
    pass
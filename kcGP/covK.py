'''
Created on Jun 29, 2015

@author: Thomas
'''

import numpy as np
import math
import scipy.spatial.distance as spdist

class Kernel(object):
    '''
    This is a base class of Kernel functions
    there is no computation in this class
    '''
    def __init__(self):
        '''
        Constructor
        '''
        self.hyp = []
        self.para = []
    
    def getCovMatrix(self, x=None, z=None, mode=None):
        '''
        Return the specific covariance matrix according to input mode
        
        :param x: training data
        :param z: test data
        :param str mode: 'self_test' return self covariance matrix of test data(test by 1).
                         'train' return training covariance matrix(train by train).
                         'cross' return cross covariance matrix between x and z(train by test)
        
        :return: the corresponding covariance matrix
        '''
        pass
    
    def getDerMatrix(self, x=None, z=None, mode=None, der=None):
        '''
        Compute derivatives wrt. hyperparameters according to input mode

        :param x: training data
        :param z: test data
        :param str mode: 'self_test' return self derivative matrix of test data(test by 1).
                         'train' return training derivative matrix(train by train).
                         'cross' return cross derivative matrix between x and z(train by test)
        :param int der: index of hyperparameter whose derivative to be computed

        :return: the corresponding derivative matrix
        '''
        pass
    
    def checkInputGetCovMatrix(self,x,z,mode):
        '''
        Check validity of inputs for the method getCovMatrix()
        
        :param x: training data
        :param z: test data
        :param str mode: 'self_test' return self derivative matrix of test data(test by 1).
                         'train' return training derivative matrix(train by train).
                         'cross' return cross derivative matrix between x and z(train by test)
        '''
        if mode is None:
            raise Exception("Specify the mode: 'train' or 'cross'")
        if x is None and z is None:
            raise Exception("Specify at least one: training input (x) or test input (z) or both.")
        if mode == 'cross':
            if x is None or z is None:
                raise Exception("Specify both: training input (x) and test input (z) for cross covariance.")
    
    def checkInputGetDerMatrix(self,x,z,mode,der):
        '''
        Check validity of inputs for the method getDerMatrix()
        
        :param x: training data
        :param z: test data
        :param str mode: 'self_test' return self derivative matrix of test data(test by 1).
                         'train' return training derivative matrix(train by train).
                         'cross' return cross derivative matrix between x and z(train by test)
        :param int der: index of hyperparameter whose derivative to be computed
        '''
        if mode is None:
            raise Exception("Specify the mode: 'train' or 'cross'")
        if x is None and z is None:
            raise Exception("Specify at least one: training input (x) or test input (z) or both.")
        if mode == 'cross':
            if x is None or z is None:
                raise Exception("Specify both: training input (x) and test input (z) for cross covariance.")
        if der is None:
            raise Exception("Specify the index of parameters of the derivatives.")
    
    
    # overloading
    def __add__(self,cov):
        '''
        Overloading + operator.

        :param cov: covariance function
        :return: an instance of SumOfKernel
        '''
        return SumOfKernel(self,cov)



    # overloading
    def __mul__(self,other):
        '''
        Overloading * operator.
        Using * for both multiplication with scalar and product of kernels
        depending on the type of the two objects.

        :param other: covariance function as product or int/float as scalar
        :return: an instance of ScaleOfKernel or ProductOfKernel
        '''
        if isinstance(other, int) or isinstance(other, float):
            return ScaleOfKernel(self,other)
        elif isinstance(other, Kernel):
            return ProductOfKernel(self,other)
        else:
            print "only numbers and Kernels are supported operand types for *"



    # overloading
    __rmul__ = __mul__



class ProductOfKernel(Kernel):
    '''
    Product of two kernel functions
    '''
    def __init__(self,cov1,cov2):
        self.cov1 = cov1
        self.cov2 = cov2
        self._hyp = cov1.hyp + cov2.hyp

    def _setHyp(self,hyp):
        assert len(hyp) == len(self._hyp)
        len1 = len(self.cov1.hyp)
        self._hyp = hyp
        self.cov1.hyp = self._hyp[:len1]
        self.cov2.hyp = self._hyp[len1:]
    def _getHyp(self):
        return self._hyp
    hyp = property(_getHyp,_setHyp)

    def getCovMatrix(self,x=None,z=None,mode=None):
        self.checkInputGetCovMatrix(x,z,mode)
        A = self.cov1.getCovMatrix(x,z,mode) * self.cov2.getCovMatrix(x,z,mode)
        return A

    def getDerMatrix(self,x=None,z=None,mode=None,der=None):
        self.checkInputGetDerMatrix(x,z,mode,der)
        if der < len(self.cov1.hyp):
            A = self.cov1.getDerMatrix(x,z,mode,der) * self.cov2.getCovMatrix(x,z,mode)
        elif der < len(self.hyp):
            der2 = der - len(self.cov1.hyp)
            A = self.cov2.getDerMatrix(x,z,mode,der2) * self.cov1.getCovMatrix(x,z,mode)
        else:
            raise Exception("Error: der out of range for covProduct")
        return A


class SumOfKernel(Kernel):
    '''
    Sum of two kernel functions
    '''
    def __init__(self,cov1,cov2):
        self.cov1 = cov1
        self.cov2 = cov2
        self._hyp = cov1.hyp + cov2.hyp
    def _setHyp(self,hyp):
        assert len(hyp) == len(self._hyp)
        len1 = len(self.cov1.hyp)
        self._hyp = hyp
        self.cov1.hyp = self._hyp[:len1]
        self.cov2.hyp = self._hyp[len1:]
    def _getHyp(self):
        return self._hyp
    hyp = property(_getHyp,_setHyp)

    def getCovMatrix(self,x=None,z=None,mode=None):
        self.checkInputGetCovMatrix(x,z,mode)
        A = self.cov1.getCovMatrix(x,z,mode) + self.cov2.getCovMatrix(x,z,mode)
        return A

    def getDerMatrix(self,x=None,z=None,mode=None,der=None):
        self.checkInputGetDerMatrix(x,z,mode,der)
        if der < len(self.cov1.hyp):
            A = self.cov1.getDerMatrix(x,z,mode,der)
        elif der < len(self.hyp):
            der2 = der - len(self.cov1.hyp)
            A = self.cov2.getDerMatrix(x,z,mode,der2)
        else:
            raise Exception("Error: der out of range for covSum")
        return A


class ScaleOfKernel(Kernel):
    '''
    Scale of a kernel function
    '''
    def __init__(self,cov,scalar):
        self.cov = cov
        if cov.hyp:
            self._hyp = [scalar] + cov.hyp
        else:
            self._hyp = [scalar]
    def _setHyp(self,hyp):
        assert len(hyp) == len(self._hyp)
        self._hyp = hyp
        self.cov.hyp = self._hyp[1:]
    def _getHyp(self):
        return self._hyp
    hyp = property(_getHyp,_setHyp)

    def getCovMatrix(self,x=None,z=None,mode=None):
        self.checkInputGetCovMatrix(x,z,mode)
        sf2 = np.exp(self.hyp[0])                     # scale parameter
        A = sf2 * self.cov.getCovMatrix(x,z,mode)     # accumulate cov
        return A

    def getDerMatrix(self,x=None,z=None,mode=None,der=None):
        self.checkInputGetDerMatrix(x,z,mode,der)
        sf2 = np.exp(self.hyp[0])                     # scale parameter
        if der == 0:                                  # compute derivative w.r.t. sf2
            A = 2. * sf2 * self.cov.getCovMatrix(x,z,mode)
        else:
            A = sf2 * self.cov.getDerMatrix(x,z,mode,der-1)
        return A


class RBF(Kernel):
    '''
    Squared Exponential kernel with isotropic distance measure. hyp = [log_ell, log_sigma]
    
    :param: log_ell: characteristic length scale
    :param: log_sigma: signal deviation
    '''
    def __init__(self, log_ell=0., log_sigma=0.):
        self.hyp = [log_ell, log_sigma]
    
    def getCovMatrix(self, x=None, z=None, mode=None):
        self.checkInputGetCovMatrix(x, z, mode)
        ell = np.exp(self.hyp[0])
        sf2 = np.exp(2*self.hyp[1])
        
        if mode == 'self_test':         # self covariances for the test cases
            nn,D = z.shape
            A = np.zeros((nn,1))
        elif mode == 'train':           # compute covariance matrix for training set
            A = spdist.cdist(x/ell,x/ell,'sqeuclidean')
        elif mode == 'cross':           # compute covariance between data sets x and z
            A = spdist.cdist(x/ell,z/ell,'sqeuclidean')
        A = sf2 * np.exp(-0.5*A)
        
        return A
    
    def getDerMatrix(self, x=None, z=None, mode=None, der=None):
        self.checkInputGetDerMatrix(x, z, mode, der)
        ell = np.exp(self.hyp[0])
        sf2 = np.exp(2*self.hyp[1])
        
        if mode == 'self_test':         # self covariances for the test cases
            nn,D = z.shape
            A = np.zeros((nn,1))
        elif mode == 'train':           # compute covariance matrix for training set
            A = spdist.cdist(x/ell,x/ell,'sqeuclidean')
        elif mode == 'cross':           # compute covariance between data sets x and z
            A = spdist.cdist(x/ell,z/ell,'sqeuclidean')
        
        if der == 0:                    # compute derivative matrix wrt 1st parameter
            A = sf2 * np.exp(-0.5*A) * A
        elif der == 1:                  # compute derivative matrix wrt 2nd parameter
            A = 2. * sf2 * np.exp(-0.5*A)
        else:
            raise Exception('Calling for a derivative in RBF that does not exist')
        
        return A
    
    
class Pre(Kernel):
    '''
    Precomputed kernel matrix 
    No hyperparameters and thus nothing will be optimized.
    
    :param M1: cross covariances matrix(train+1 by test).
               last row is self covariances (diagonal of test by test)
    :param M2: training set covariance matrix (train by train)
    '''
    def __init__(self,M1,M2):
        self.M1 = M1
        self.M2 = M2
        self.hyp = []
    
    
    def getCovMatrix(self, x=None, z=None, mode=None):
        if mode == 'self_test':         # diagonal covariance between test_test
            A = self.M1[-1,:]           # self covariances for the test cases (last row)
            A = np.reshape(A, (A.shape[0],1))
        elif mode == 'train':
            A = self.M2
        elif mode == 'cross':
            A = self.M1[:-1,:]
            
        return A
    
    def getDerMatrix(self, x=None, z=None, mode=None, der=None):
        if not der is None:
            raise Exception('Error: NO optimization in precomputed kernel matrix')
        
        return 0

if __name__ == '__main__':
    pass
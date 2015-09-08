'''
Created on Jul 5, 2015

@author: Thomas
'''
from __future__ import division
import numpy as np
import scipy.stats as ss

class Joint_dist(object):
    '''
    Joint distribution class
    
    A practice for understanding MCMC (M-H algo., Slice sampling & Elliptical slice sampling)
    '''    
    def pdf(self, sample):
        '''
        Get the probability of a specific sample
        '''
        #v = sample[0]
        #pv = ss.norm(0, 3).pdf(v)
        #xs = sample[0:]
        pxs = [ss.norm(0, 0.1).pdf(x_k) for x_k in sample]     #np.sqrt(np.e)
        return np.array(pxs)
    
    def loglike(self, sample):
        '''
        log likelihood of a specific sample
        '''
        #print np.log(self.pdf(sample))
        return np.sum(np.log(self.pdf(sample)))
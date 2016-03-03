'''
Created on Sep 10, 2015

@author: Thomas
'''
import csv
import kcGP
import timeit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from kcMCMC import sdsK

def mcmc_update(data, n_samples, ll=3.0, sf=15.0, sn=0.5):
    x = data['x']
    y = data['y']
    hyp = np.array([ll, sf, sn])
    f = np.zeros_like(y)

    histHyp = np.zeros((3, n_samples))
    histF = np.zeros((f.shape[0], n_samples))
    for s in range(n_samples):
        print '==========Iteration %d==========' %(s+1)
        propF, propHyp = sdsK.surrogate_slice_sampling(f, x, y, hyp, scale=np.asarray([8., 8., 5.]))
        propF = sdsK.elliptical_slice(propF, x, y, propHyp)
        print 'll=%.3f, sf=%.3f, sn=%.3f' %(propHyp[0], propHyp[1], propHyp[2])
        print 'f\'s: %.3f, %.3f' %(propF[0], propF[1])
        f = propF
        hyp = propHyp

        histHyp[:, s] = propHyp
        histF[:, s] = propF

    return histHyp, histF

def wrapper(func, *args, **kwargs):
    """Wrapper function for timeit.time use

    Parameters
    ----------
    func: Python function
        target function to be measured
    args: non-keyworded argument
        list of non-keyworded arguments
    kwargs: keyworded argument
        dict of keyworded arguments
    """
    def wrapped():
        return func(*args, **kwargs)

    return wrapped

def output_result(x, y, histF, histHyp):
    """
    Parameters
    ----------
    x: training x
    y: training y
    histF: proposed latent variables f's
    histHyp: proposed hyper-parameters
    """
    n_samples = histF.shape[1]
    with open('./output/ess_F.csv', 'wb') as f:
        writer = csv.writer(f)
        first_row = range(1, n_samples+1)
        first_row.append("x")
        first_row.append("y")
        writer.writerow(first_row)
        xy = np.hstack((x, y))
        writer.writerows(np.hstack((histF, xy)))

    with open('./output/ess_hyp.csv', 'wb') as h:
        writer = csv.writer(h)
        writer.writerow(["ll", "sf2", "sn"])
        writer.writerows(histHyp.T)

if __name__ == "__main__":
    demoData = np.load('./output/regression_data.npz')
    x = demoData['x']
    y = demoData['y']
    y = y.reshape((y.shape[0],))
    xs= demoData['xstar']

    synthetic = pd.read_csv('./output/synthetic.csv')

    model = kcGP.gpK.GPR()
    findOPT = raw_input("Optimize (o) or MCMC (m)? ")
    if findOPT == "o":
        model.optimize(x, y)
        print 'll %.4f' %(np.exp(model.covfunc.hyp[0]))
        print 'sf %.4f' %(np.exp(model.covfunc.hyp[1]))
        print 'sn %.4f' %(np.exp(model.likfunc.hyp[0]))
    
    elif findOPT == "m":
        idx = np.argsort(x.reshape((x.shape[0],)))
        x = np.sort(x, axis=0)
        y = y[idx]
        y[1] = 0.

        x = synthetic['x']
        x = x.reshape((x.shape[0], 1))
        y = synthetic['y']

        data = {'x': x, 'y': y}
        num_iters = 5000
        print '# of iterations:', num_iters
        mcmc = wrapper(mcmc_update, data, num_iters)
        # print 'Time needed:', timeit.timeit(mcmc, number=1)

        histHyp, histF = mcmc()
        y = y.reshape((y.shape[0], 1))
        output_result(x, y, histF, histHyp)

    else:
        np.random.seed(124)
        ll = 5.0
        sf = 20.0
        sn = 2.5
        x = np.arange(0, 455)

        covMCMC = kcGP.covK.RBF(np.log(ll), np.log(sf))
        K = covMCMC.getCovMatrix(x.reshape((x.shape[0], 1)), mode='train')
        L = kcGP.tools.jitchol(K+sn**2*np.eye(K.shape[0]))

        z = np.random.normal(size=(K.shape[0],))

        # print np.exp(covMCMC.hyp[0]), np.exp(covMCMC.hyp[1])
        y = np.fmax(np.fmin(np.dot(L,z)+91.1538461538, 100), 0)
        f = np.dot(L,z)+91.1538461538

        plt.figure()
        plt.plot(x, f, label='f', color='green')
        plt.plot(x, y, label='y', color='blue', ls='None', marker='x', ms=6, mew=2)
        plt.title('Synthetic data with ll=%r, sf=%r, sn=%r' %(ll, sf, sn))
        plt.xlabel('input x')
        plt.ylabel('synthetic y/f')
        plt.legend()
        plt.show()

        # output = np.vstack((np.vstack((f, y)), x))
        # with open('./output/synthetic.csv', 'wb') as h:
        #     writer = csv.writer(h)
        #     writer.writerow(["f", "y", "x"])
        #     writer.writerows(output.T)

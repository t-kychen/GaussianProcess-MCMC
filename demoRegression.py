'''
Created on Sep 10, 2015

@author: Thomas
'''
import sys
import csv
import kcGP
import pyGPs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from kcMCMC import sliceSample

def mcmcUpdate(data, n_samples, ll=0.35, sf=2.0, sn=0.2):
    x = data['x']
    y = data['y']
    propHyp = np.array([ll, sf, sn])
    propF = np.zeros_like(y)

    histHyp = np.zeros((3, n_samples))
    histF = np.zeros((propF.shape[0], n_samples))
    for s in range(n_samples):
        print '==========Iteration %d==========' %(s+1)
        propF, propHyp = sliceSample.surrogate_slice_sampling(propF, x, y, propHyp, scale=np.asarray([10., 10., 5.]), iter=s)
        # propF = sdsK.elliptical_slice(propF, x, y, propHyp)
        print 'll=%.3f, sf=%.3f, sn=%.3f' %(propHyp[0], propHyp[1], propHyp[2])

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

def outputDemoRes(x, y, histF, histHyp):
    """
    Parameters
    ----------
    x: training x
    y: training y
    histF: proposed latent variables f's
    histHyp: proposed hyper-parameters
    """
    n_samples = histF.shape[1]
    with open('./output/demo_f.csv', 'wb') as f:
        writer = csv.writer(f)
        first_row = range(1, n_samples+1)
        first_row.append("x")
        first_row.append("y")
        writer.writerow(first_row)
        xy = np.hstack((x, y))
        writer.writerows(np.hstack((histF, xy)))

    with open('./output/demo_hyp.csv', 'wb') as h:
        writer = csv.writer(h)
        writer.writerow(["ll", "sf2", "sn"])
        writer.writerows(histHyp.T)

if __name__ == "__main__":

    num_iters = int(sys.argv[1])
    model = kcGP.gpK.GPR()

    dataOption = raw_input("Toy example (t) or synthetic data (s)? ")

    if dataOption == "t":
        # toy example
        demoData = np.load('./output/regression_data.npz')
        x = demoData['x']
        y = demoData['y']
        y = y.reshape((y.shape[0],))
        xs= demoData['xstar']
        idx = np.argsort(x.reshape((x.shape[0],)))
        x = np.sort(x, axis=0)
        y = y[idx]
        y[1] = 0.
    elif dataOption == "s":
        # synthetic condition score generated from GP
        synthetic = pd.read_csv('./output/synthetic.csv')
        x = synthetic['x']
        x = x.reshape((x.shape[0], 1))
        y = synthetic['y']

    runOption = raw_input("Optimize (o) or MCMC (m)? ")
    if runOption == "m":
        data = {'x': x, 'y': y}
        mcmc = wrapper(mcmcUpdate, data, num_iters)
        # print 'Time needed:', timeit.timeit(mcmc, number=1)

        histHyp, histF = mcmc()
        y = y.reshape((y.shape[0], 1))
        outputDemoRes(x, y, histF, histHyp)

    elif runOption == "o":
        model = pyGPs.GPR()
        model.getPosterior(x, y)
        model.optimize(x, y)
        model.predict(xs)
        model.plot()
    else:
        # generate synthetic data
        np.random.seed(124)
        ll = 5.0
        sf = 20.0
        sn = 2.5
        x = np.arange(0, 455)

        covMCMC = kcGP.covK.RBF(np.log(ll), np.log(sf))
        K = covMCMC.getCovMatrix(x.reshape((x.shape[0], 1)), mode='train')
        L = kcGP.tools.jitchol(K+sn**2*np.eye(K.shape[0]))

        z = np.random.normal(size=(K.shape[0],))
        y = np.dot(L,z)+91.1538461538
        f = np.dot(L,z)+91.1538461538

        output = np.vstack((np.vstack((f, y)), x))
        with open('./output/synthetic.csv', 'wb') as h:
            writer = csv.writer(h)
            writer.writerow(["f", "y", "x"])
            writer.writerows(output.T)


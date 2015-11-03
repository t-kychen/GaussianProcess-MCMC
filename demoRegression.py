'''
Created on Sep 10, 2015

@author: Thomas
'''
import os
import csv
import kcGP
import numpy as np
from MCMC import surrogate_slice_sampling, elliptical_slice, infMCMC

def outputResult(x, y, histF, histHyp):
    '''
    :param x: training x
    :param y: training y
    :param histF: proposed latent variables f's
    :param histHyp: proposed hyper-parameters

    :return: None. Instead, output proposed f's and hyper-parameters
    '''
    n_samples = histF.shape[0]
    cwd = os.getcwd()
    with open(cwd+'/output/ess_F.csv', 'wb') as f:
        writer = csv.writer(f)
        first_row = range(1, n_samples+1)
        first_row.append("x")
        first_row.append("y")
        writer.writerow(first_row)
        xy = np.hstack((x, y))
        writer.writerows(np.hstack((histF, xy)))

    with open(cwd+'/output/ess_hyp.csv', 'wb') as h:
        writer = csv.writer(h)
        writer.writerow(["ll", "sf2", "sn"])
        writer.writerows(histHyp.T)


if __name__ == "__main__":
    demoData = np.load('regression_data.npz')
    x = demoData['x']               # training data
    y = demoData['y']               # training target
    y = y.reshape((y.shape[0],))
    xs= demoData['xstar']           # test data

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
        hyp = np.asarray([0.5, 2.5])        # true values: 0.370316630438, 1.9856442896249067
        sn = 0.3                            # true value: 0.149188184796
        f = np.zeros_like(y)

        var = np.asarray([f, x, y, hyp])
        n_samples = 10000
        histHyp = np.zeros((hyp.shape[0]+1, n_samples))
        histF = np.zeros((f.shape[0], n_samples))
        histLlk = []
        
        for s in range(n_samples):
            print "===Iteration %d===" %(s+1)
            propF, propHyp = surrogate_slice_sampling(var=var, sn=sn, scale=np.asarray([2.5, 2.5, 1.]), opt=0)
            var[0] = propF
            var[3] = propHyp[0:2]
            sn = propHyp[2]

            histHyp[:, s] = propHyp
            histF[:, s] = propF

        y = y.reshape((y.shape[0], 1))
        outputResult(x, y, histF, histHyp)

        covMCMC = kcGP.covK.RBF(np.log(var[3][0]), np.log(var[3][1]))
        model.setPrior(kernel=covMCMC)
        model.setNoise(np.log(sn))
        model.getPosterior(x, y)

        model.xs = xs
        ym, ys_lw, ys_up, fmu, fs2 = infMCMC(xs, histF[:, -500:], model)
        model.ym = ym + np.mean(y)
        model.plot(ys_lw+np.mean(y), ys_up+np.mean(y))
'''
Created on Sep 15, 2015

@author: Thomas
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def traceHyp(data, colors, iter_mcmc, fold):
    '''Plot the trace of hyper-parameters'''
    plt.subplot(311)
    plt.title("Traces of hyper parameters")
    plt.plot(data["ll"][iter_mcmc*fold:iter_mcmc*(fold+1)], label="$ll$", c=colors[0], lw=2)
    plt.legend(loc="upper right")
    plt.grid()

    plt.subplot(312)
    plt.plot(data["sf2"][iter_mcmc*fold:iter_mcmc*(fold+1)], label=r'$\sigma_y$', c=colors[1], lw=2)
    plt.legend(loc="upper right")
    plt.grid()

    plt.subplot(313)
    plt.plot(data["sn"][iter_mcmc*fold:iter_mcmc*(fold+1)], label=r"$\sigma_n$", c=colors[2], lw=2)
    plt.legend(loc="upper right")
    plt.grid()

    plt.xlabel("Iterations")
    plt.show()


def histHyp(data, colors, iter_mcmc, fold):
    '''Plot the histogram of hyper-parameters'''
    plt.subplot(311)
    plt.title("Posterior of hyper-parameters")
    plt.hist(data["ll"][iter_mcmc*fold:iter_mcmc*(fold+1)], color=colors[0], bins=300, label="$ll$", histtype="bar")
    print 'mean of ll ', np.mean(data["ll"])
    plt.legend(loc="upper right")
    plt.grid()

    plt.subplot(312)
    plt.hist(data["sf2"][iter_mcmc*fold:iter_mcmc*(fold+1)], color=colors[1], bins=300, label=r"$\sigma_y$", histtype="bar")
    print 'mean of sigma y ', np.mean(data["sf2"])
    plt.legend(loc="upper right")
    plt.grid()
        
    plt.subplot(313)
    plt.hist(data["sn"][iter_mcmc*fold:iter_mcmc*(fold+1)], color=colors[2], bins=300, label=r"$\sigma_n$", histtype="bar")
    plt.legend(loc="upper right")
    plt.grid()
    plt.show()


def plotFY(x, y, f):
    '''plot f and y'''
    plt.title("Proposed f and actual y")
    plt.scatter(x, y, color="#348ABD")
    plt.plot(x, f, color="#467821", lw=2)
    plt.xlim(x.min()-1, x.max()+1)
    plt.ylim(y.min()-1, y.max()+1)
    plt.grid()
    plt.show()


def traceF(loc, y, f):
    '''plot f at specified location'''
    y = np.zeros_like(f) + y
    plt.title("Trace of f at x=%.2f" %loc)
    plt.plot(y, color="#348ABD", lw=2)
    plt.plot(f, color="#467821", lw=2)
    plt.grid()
    plt.show()
    
    
def multiPlot(data, name):
    '''Plot histogram'''
    plt.subplot(211)
    plt.title("%s" %name)
    plt.hist(data[:,0], color="#348ABD", bins=50, histtype="bar")
    plt.xlim(0,100)
    plt.grid()
    
    plt.subplot(212)
    plt.plot(data[:,1], data[:,0], color="#348ABD", lw=2)
    plt.xlim(data[:,1].min(), data[:,1].max())
    plt.ylim(data[:,0].min(), data[:,0].max()+10)
    plt.grid()
    plt.show()

    
if __name__ == "__main__":
    cwd = os.getcwd()
    colors = ["#348ABD", "#A60628", '#467821']
    # hyp_llk = pd.read_csv("Surr_hyper_llk.csv").T
    # hyp_llk.columns = ["ll", "sf2", "sn2"]           # set up column names
    #
    # # plot trace of hyper parameters
    # traceHyp(hyp_llk, colors)
    #
    # # plot histogram of hyper-parameters
    # histHyp(hyp_llk, colors)
    
    # fy = pd.read_csv(cwd+"/output/Proposed_F.csv")
    # for col in range(-6, -2):
    #     plotFY(fy["x"], fy["y"], fy[fy.columns[col]])
    #
    # for loc in range(15, 20):
    #     traceF(loc=fy.T[loc][-2], y=fy.T[loc][-1], f=fy.T[loc][:-2])

    iter_mcmc = 2000
    fold = 0
    hyp = pd.read_csv(cwd+"/output/hyp_gap0.5.csv")
    histHyp(hyp, colors, iter_mcmc, fold)
    traceHyp(hyp, colors, iter_mcmc, fold)
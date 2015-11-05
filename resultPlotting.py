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
    
    
def cvLLK(llk, iter_mcmc):
    '''
    Plot llk's in 10 folds and the average of them
    :param llk: Pandas, llk data
    :return: None
    '''
    plt.plot(llk['avg'], marker='o', markersize=5, label='log likelihood (%s iters)' %iter_mcmc)
    plt.xticks(range(llk['avg'].shape[0]), ['gap 0.5', 'gap 1', 'gap 2', 'gap 3', 'gap 4'])
    for i in range(llk['avg'].shape[0]):
        plt.text(i, llk['avg'][i]+5, llk['avg'][i])
    plt.margins(0.2)
    plt.grid()
    plt.legend(fontsize=12)
    plt.show()


if __name__ == "__main__":
    cwd = os.getcwd()
    colors = ["#348ABD", "#A60628", '#467821']
    iter_mcmc = 2000
    fold = 0

    # fy = pd.read_csv(cwd+"/output/Proposed_F.csv")
    # for col in range(-6, -2):
    #     plotFY(fy["x"], fy["y"], fy[fy.columns[col]])
    #
    # for loc in range(15, 20):
    #     traceF(loc=fy.T[loc][-2], y=fy.T[loc][-1], f=fy.T[loc][:-2])

    # hyp = pd.read_csv(cwd+"/output/hyp_gap0.5.csv")
    # histHyp(hyp, colors, iter_mcmc, fold)
    # traceHyp(hyp, colors, iter_mcmc, fold)

    llk = pd.read_csv(cwd+"/output/llk.csv", usecols=['f1','f2','f3','f4','f5','f6','f7','f8','f9','f10'])
    llk['avg'] = llk.mean(axis=1)
    cvLLK(llk, iter_mcmc)
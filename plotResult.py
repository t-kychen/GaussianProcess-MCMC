"""
    File name: plotResult.py
    Author: Kuan-Yu Chen
    Python version: 2.7
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import kcGP
import matplotlib.cm as cm
from kcMCMC import sliceSample

def trace_hyp(data0, colors, iter_mcmc, fold, data1=None, data2=None, data3=None, data4=None):
    '''Plot the trace of hyper-parameters'''
    plt.subplot(311)
    plt.title("Trace of hyper-parameters")
    plt.plot(data0["ll"][iter_mcmc*fold:iter_mcmc*(fold+1)], label=r'$ll$', c=colors[0], lw=1)
    if not (data1 is None):
        plt.plot(data1["sn"][iter_mcmc*fold:iter_mcmc*(fold+1)], label="gap 1", c=colors[1], lw=1)
        plt.plot(data2["sn"][iter_mcmc*fold:iter_mcmc*(fold+1)], label="gap 2", c=colors[2], lw=1)
        plt.plot(data3["sn"][iter_mcmc*fold:iter_mcmc*(fold+1)], label="gap 3", c=colors[3], lw=1)
        plt.plot(data4["sn"][iter_mcmc*fold:iter_mcmc*(fold+1)], label="gap 4", c=colors[4], lw=1)
    plt.legend(loc="upper right", fontsize=12)
    plt.grid()

    plt.subplot(312)
    plt.plot(data0["sf2"][iter_mcmc*fold:iter_mcmc*(fold+1)], label=r'$\sigma_y$', c=colors[0], lw=2)
    if not (data1 is None):
        plt.plot(data1["sf2"][iter_mcmc*fold:iter_mcmc*(fold+1)], label=r'$\sigma_y$', c=colors[1], lw=2)
        plt.plot(data2["sf2"][iter_mcmc*fold:iter_mcmc*(fold+1)], label=r'$\sigma_y$', c=colors[2], lw=2)
        plt.plot(data3["sf2"][iter_mcmc*fold:iter_mcmc*(fold+1)], label=r'$\sigma_y$', c=colors[3], lw=2)
        plt.plot(data4["sf2"][iter_mcmc*fold:iter_mcmc*(fold+1)], label=r'$\sigma_y$', c=colors[4], lw=2)
    plt.legend(loc="upper right", fontsize=12)
    plt.grid()

    plt.subplot(313)
    plt.plot(data0["sn"][iter_mcmc*fold:iter_mcmc*(fold+1)], label=r"$\sigma_n$", c=colors[0], lw=2)
    if not (data1 is None):
        plt.plot(data1["sn"][iter_mcmc*fold:iter_mcmc*(fold+1)], label=r"$\sigma_n$", c=colors[1], lw=2)
        plt.plot(data2["sn"][iter_mcmc*fold:iter_mcmc*(fold+1)], label=r"$\sigma_n$", c=colors[2], lw=2)
        plt.plot(data3["sn"][iter_mcmc*fold:iter_mcmc*(fold+1)], label=r"$\sigma_n$", c=colors[3], lw=2)
        plt.plot(data4["sn"][iter_mcmc*fold:iter_mcmc*(fold+1)], label=r"$\sigma_n$", c=colors[4], lw=2)
    plt.legend(loc="upper right", fontsize=12)
    plt.grid()

    plt.xlabel("Iterations")
    plt.show()

def hist_hyp(data0, colors, data1=None, data2=None, data3=None, data4=None):
    '''Plot the histogram of hyper-parameters'''
    plt.subplot(311)
    plt.title("Posterior of hyper-parameters")
    plt.hist(data0[:, 0], color=colors[0], alpha=0.7, bins=300, label=r"$l l$", histtype="bar")
    plt.legend(loc="upper right", fontsize=12)
    plt.grid()

    # print 'largest x', np.mean(hyp)
    plt.subplot(312)
    plt.hist(data0[:, 1], color=colors[1], alpha=0.7, bins=300, label=r"$\sigma_y$", histtype="bar")
    plt.legend(loc="upper right")
    plt.grid()

    plt.subplot(313)
    plt.hist(data0[:, 2], color=colors[2], alpha=0.7, bins=300, label=r"$\sigma_n$", histtype="bar")
    plt.legend(loc="upper right")
    plt.grid()
    plt.show()

def plot_fy(x, y, f, loc=None):
    '''plot f at specified location'''
    y = np.zeros(f.shape[0]) + y
    if not (loc is None):
        plt.title("Trace of f at x=%.2f" %loc)

    plt.plot(x, y, color="#348ABD", ls='None', marker='+', ms=8, mew=2)
    if len(f.shape) > 1:
        color_map = iter(cm.rainbow(np.linspace(0, 1, f.shape[1])))
        for i in range(f.shape[1]):
            plt.plot(x, f[:, i], color=next(color_map), lw=2)
    else:
        plt.plot(x, f, color="#467821", lw=2)
    plt.grid()
    plt.show()

def inference_plot(hist_f, hist_hyp, iter_mcmc):
    '''
    Inference from proposed data
    :param hist_f: Pandas, proposed f's
    :param hist_hyp:  Pandas, proposed hyp's
    :return: None
    '''
    x = hist_f['x']
    y = hist_f['y']
    upper = 100 - np.mean(y)
    lower = 0 - np.mean(y)

    hist_f = hist_f.drop(['x', 'y'], axis=1)
    selected_sample = np.arange(iter_mcmc*9/10-1, iter_mcmc, 10)

    model = kcGP.gpK.GPR()
    xs = np.arange(np.min(x), np.max(x), 0.05)
    xs = np.reshape(xs, (xs.shape[0], 1))
    model.xs = xs

    ll = np.mean(hist_hyp['ll'][selected_sample])
    sf = np.mean(hist_hyp['sf2'][selected_sample])
    sn = np.mean(hist_hyp['sn'][selected_sample])
    print 'Hyper-parameters for inference:'
    print 'll: %.2f, sf: %.2f, sn: %.2f \n' %(ll, sf, sn)

    hist_f = np.mean(hist_f.loc[:, selected_sample.astype('str')], axis=1).reshape((hist_f.shape[0], 1))

    covMCMC = kcGP.covK.RBF(np.log(ll), np.log(sf))
    model.setPrior(kernel=covMCMC)
    model.setNoise(np.log(sn))
    model.setData(x, y)

    trunclik = kcGP.likK.TruncatedGauss2(upper=upper, lower=lower, log_sigma=np.log(sn))
    model.likfunc = trunclik

    ym, ys_lw, ys_up, _ = sliceSample.inf_mcmc(hist_f, model)
    model.ym = ym
    model.plot(ys_lw, ys_up)

if __name__ == "__main__":
    colors = ["#348ABD", '#FF0000', '#228B22', '#FFD700', '#FF6347']
    fold = 0

    f = pd.read_csv("./output/0407/fGap1.csv")
    hyp = pd.read_csv("./output/0407/hypGap1.csv")
    iter_mcmc = f.shape[1] - 2  # Extra 2 columns are x and y

    trace_hyp(hyp, colors, iter_mcmc, fold)
    hist_hyp(np.asarray(hyp.iloc[501:iter_mcmc]), colors)

    # last F
    plot_fy(x=f['x'], y=f['y'], f=np.array(f.loc[:, str(iter_mcmc)])+np.mean(f['y']))
    # rainbow plot
    plot_fy(x=f['x'], y=f['y'], f=np.array(f.loc[:, str(iter_mcmc-500):str(iter_mcmc)]+np.mean(f['y'])))

    # inference from MCMC results and make prediction plot
    inference_plot(f, hyp, iter_mcmc)


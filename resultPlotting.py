'''
Created on Sep 15, 2015

@author: Kuan-Yu Chen
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import kcGP
import matplotlib.cm as cm
from kcMCMC import sdsK

def trace_hyp(data0, colors, iter_mcmc, fold, data1=None, data2=None, data3=None, data4=None):
    '''Plot the trace of hyper-parameters'''
    plt.subplot(311)
    plt.title("Traces of length scale")
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

def cvTraceHyp(data, iter_mcmc, fold, num):
    '''Plot the trace of cv hyper-parameters'''
    hyp = ['ll', 'sf2', 'sn']
    names = ['length scale', 'magnitude', 'noise']
    plt.title("Traces of %s, gap = 4" %(names[num]))
    plt.plot(data[hyp[num]][iter_mcmc*fold:iter_mcmc*(fold+1)], label="fold 1", lw=1)
    plt.plot(data[hyp[num]][iter_mcmc*(fold+1):iter_mcmc*(fold+2)], label="fold 2", lw=1)
    plt.plot(data[hyp[num]][iter_mcmc*(fold+2):iter_mcmc*(fold+3)], label="fold 3", lw=1)
    plt.plot(data[hyp[num]][iter_mcmc*(fold+3):iter_mcmc*(fold+4)], label="fold 4", lw=1)
    plt.plot(data[hyp[num]][iter_mcmc*(fold+4):iter_mcmc*(fold+5)], label="fold 5", lw=1)
    plt.plot(data[hyp[num]][iter_mcmc*(fold+5):iter_mcmc*(fold+6)], label="fold 6", lw=1)
    plt.plot(data[hyp[num]][iter_mcmc*(fold+6):iter_mcmc*(fold+7)], label="fold 7", lw=1)
    plt.plot(data[hyp[num]][iter_mcmc*(fold+7):iter_mcmc*(fold+8)], label="fold 8", lw=1)
    plt.plot(data[hyp[num]][iter_mcmc*(fold+8):iter_mcmc*(fold+9)], label="fold 9", lw=1)
    plt.plot(data[hyp[num]][iter_mcmc*(fold+9):iter_mcmc*(fold+10)], label="fold 10", lw=1)
    plt.legend(loc="upper right", fontsize=12)
    plt.grid()
    plt.xlabel("Iterations")
    plt.show()

def hist_hyp(data0, colors, data1=None, data2=None, data3=None, data4=None):
    '''Plot the histogram of hyper-parameters'''
    plt.subplot(311)
    plt.title("Posterior of length scale")
    plt.hist(data0['ll'], color=colors[0], alpha=0.7, bins=300, label=r"$l l$", histtype="bar")
    plt.legend(loc="upper right", fontsize=12)
    plt.grid()

    # print 'largest x', np.mean(hyp)
    plt.subplot(312)
    plt.hist(data0["sf2"], color=colors[1], alpha=0.7, bins=300, label=r"$\sigma_y$", histtype="bar")
    plt.legend(loc="upper right")
    plt.grid()

    plt.subplot(313)
    plt.hist(data0["sn"], color=colors[2], alpha=0.7, bins=300, label=r"$\sigma_n$", histtype="bar")
    plt.legend(loc="upper right")
    plt.grid()
    plt.show()

def plot_fy(y, f, loc=None):
    '''plot f at specified location'''
    y = np.zeros(f.shape[0]) + y
    if not (loc is None):
        plt.title("Trace of f at x=%.2f" %loc)

    plt.plot(y, color="#348ABD", lw=2)
    if len(f.shape) > 1:
        color_map = iter(cm.rainbow(np.linspace(0, 1, f.shape[1])))
        for i in range(f.shape[1]):
            plt.plot(f[:, i], color=next(color_map), lw=2)
    else:
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

def inference_plot(hist_f, hist_hyp, iter_mcmc):
    '''
    Inference from proposed data
    :param hist_f: Pandas, proposed f's
    :param hist_hyp:  Pandas, proposed hyp's
    :return: None
    '''
    selected_sample = np.arange(iter_mcmc*3/4-1, iter_mcmc, 100)

    model = kcGP.gpK.GPR()
    ll = np.mean(hist_hyp['ll'][selected_sample])
    sf = np.mean(hist_hyp['sf2'][selected_sample])
    sn = np.mean(hist_hyp['sn'][selected_sample])
    print 'Hyper-parameters for inference...'
    print 'll: %.2f, sf: %.2f, sn: %.2f' %(ll, sf, sn)

    covMCMC = kcGP.covK.RBF(np.log(ll), np.log(sf))
    model.setPrior(kernel=covMCMC)
    model.setNoise(np.log(sn))
    x = hist_f['x']
    y = hist_f['y']
    hist_f = hist_f.drop(['x', 'y'], axis=1)
    model.getPosterior(x, y)      # x, y
    xs = np.arange(np.min(x), np.max(x), 0.1)
    xs = np.reshape(xs, (xs.shape[0], 1))
    model.xs = xs

    ym, ys_lw, ys_up = sdsK.inf_mcmc(xs, hist_f.loc[:, selected_sample.astype('str')], model)
    model.ym = ym
    model.plot(ys_lw, ys_up)

if __name__ == "__main__":
    colors = ["#348ABD", '#FF0000', '#228B22', '#FFD700', '#FF6347']
    iter_mcmc = 3000
    fold = 0

    f = pd.read_csv("./output/sds_ess_f_newllk.csv")
    hyp = pd.read_csv("./output/sds_ess_hyp_newllk.csv")

    # trace_hyp(hyp, colors, iter_mcmc, fold)
    # plot_fy(y=f['y'], f=np.array(f.loc[:, str(iter_mcmc)]+np.mean(f['y'])))

    plot_fy(y=f['y'], f=np.array(f.loc[:, '2900':'3000']+np.mean(f['y'])))
    # pos_x = 1
    # plot_fy(y=f['y'][pos_x], f=f.loc[pos_x, '1':'5000']+np.mean(f['y']), loc=pos_x)

    # hist_hyp(hyp, colors)
    # inference_plot(f, hyp, iter_mcmc)

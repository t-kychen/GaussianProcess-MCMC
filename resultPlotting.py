'''
Created on Sep 15, 2015

@author: Thomas
'''
import matplotlib.pyplot as plt
import pandas as pd

def tracePlot(data, colors):
    '''
    Plot the trace of hyper-parameters
    '''
    starting_point = 0
    lw = 2

    plt.subplot(311)
    plt.title("Traces of hyper parameters")
    plt.plot(data["ll"][starting_point:], label="$ll$", c=colors[0], lw=lw)
    plt.legend(loc="upper right")
    plt.grid()

    plt.subplot(312)
    plt.plot(data["sf2"][starting_point:], label=r'$\sigma_y$', c=colors[1], lw=lw)
    plt.legend(loc="upper right")
    plt.grid()

    plt.subplot(313)
    plt.plot(data["sn2"][starting_point:], label=r"$\sigma_n$", c=colors[2], lw=lw)
    plt.legend(loc="upper right")
    plt.grid()
    
    plt.xlabel("Iterations")
    plt.show()


def histPlot(data, colors):
    '''
    Plot the histogram of hyper-parameters
    '''
    plt.subplot(311)
    plt.title("Posterior of hyper-parameters")
    plt.hist(data["ll"], color=colors[0], bins=200, label="$ll$", histtype="bar")
    plt.legend(loc="upper right")
    plt.grid()

    plt.subplot(312)
    plt.hist(data["sf2"], color=colors[1], bins=200, label=r"$\sigma_y$", histtype="bar")
    plt.legend(loc="upper right")
    plt.grid()
        
    plt.subplot(313)
    plt.hist(data["sn2"], color=colors[2], bins=200, label=r"$\sigma_n$", histtype="bar")
    plt.legend(loc="upper right")
    plt.grid()
    plt.show()

    
if __name__ == "__main__":
    hyp_llk = pd.read_csv("Proposed_hyper_llk.csv").T
    hyp_llk.columns = ["ll", "sf2", "sn2", "llk"]           # set up column names
    colors = ["#348ABD", "#A60628", '#467821']

    # plot trace of hyper parameters
    tracePlot(hyp_llk, colors)
    
    # plot histogram of hyper-parameters
    histPlot(hyp_llk, colors)
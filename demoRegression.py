'''
Created on Sep 10, 2015

@author: Thomas
'''
import kcGP
import pyGPs
import numpy as np

if __name__ == "__main__":
    print 'Loading demo data...'
    #demoData = np.load('regression_data.npz')
    demoData = np.load('classification_data.npz')
    x = demoData['x']           #training data
    y = demoData['y']           #training target
    xs= demoData['xstar']       #test data

    x1 = demoData['x1']          # x for class 1 (with label -1)
    x2 = demoData['x2']          # x for class 2 (with label +1)     
    t1 = demoData['t1']          # y for class 1 (with label -1)
    t2 = demoData['t2']          # y for class 2 (with label +1)
    p1 = demoData['p1']          # prior for class 1 (with label -1)
    p2 = demoData['p2']          # prior for class 2 (with label +1)

    
    print 'Training GP from pyGPs...'   #model from pyGPs package
    method = raw_input("Regression (r) or Classification (c)?")
    if method == "r":
        model = pyGPs.GPR()         
    elif method == "c":
        model = pyGPs.GPC()
        model.plotData_2d(x1,x2,t1,t2,p1,p2)
        k = pyGPs.cov.RBFard(log_ell_list=[0.05,0.17], log_sigma=1.)
        model.setPrior(kernel=k) 

    model.getPosterior(x, y)
    model.optimize(x, y)
    print 'Plot result of pyGPs...'
    if method == "r":
        model.predict(xs)
        model.plot()
        
    elif method == "c":
        n = xs.shape[0]
        model.predict(xs, ys=0 - np.ones((n,1)))
#         model.predict(xs)
        model.plot(x1,x2,t1,t2)
    
#     print '\nTraining GP from kcGP...'
#     modelK = kcGP.gpK.GPR()     #model from self-defined package
#     modelK.getPosterior(x, y)
#     modelK.optimize(x, y)
#     modelK.predict(xs)
#     print 'Plot result of kcGP...\n'
#     modelK.plot()
#     
#     print 'Cov hyp: ', np.exp(model.covfunc.hyp[0]), np.exp(2.*model.covfunc.hyp[1])
#     print 'Lik hyp: ', np.exp(2.*model.likfunc.hyp[0])
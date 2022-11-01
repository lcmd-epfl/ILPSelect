import qml
import numpy as np
from qml.math import cho_solve
from qml.kernels import get_local_kernel

def krr(kernel, properties, l2reg=1e-9):
    alpha = cho_solve(kernel, properties, l2reg=l2reg)
    return alpha

def get_kernel(X1, X2, charges1, charges2, sigma=1):
    K = qml.kernels.get_local_kernel(X1, X2, charges1, charges2, sigma)
    return K

def train_model(X_train, atoms_train, y_train, sigma=1, l2reg=1e-9):
    K_train = get_kernel(X_train, X_train, atoms_train, atoms_train, sigma=sigma)
    alpha_train = krr(K_train, y_train, l2reg=l2reg)
    return alpha_train

def train_predict_model(X_train, atoms_train, y_train, X_test, atoms_test, y_test, sigma=1, l2reg=1e-9):
    alpha_train = train_model(X_train, atoms_train, y_train, sigma=sigma, l2reg=l2reg)
    K_test = get_kernel(X_train, X_test, atoms_train, atoms_test)
    y_pred = np.dot(K_test, alpha_train)
    mae = np.abs(y_pred - y_test)[0]
    return mae, y_pred

def opt_hypers(X_train, atoms_train, y_train, X_test, atoms_test, y_test):
    sigmas = [10, 100, 1000]
    l2regs = [1e-10, 1e-7, 1e-4]
    
    maes = np.zeros((len(sigmas), len(l2regs)))
    for i, sigma in enumerate(sigmas):
        for j, l2reg in enumerate(l2regs):
            mae, y_pred = train_predict_model(X_train, atoms_train, y_train, X_test, atoms_test,
                                             y_test, sigma=sigma, l2reg=l2reg)
            print('sigma', sigma, 'l2reg', l2reg, 'mae', mae)
            maes[i,j] = mae
            
    min_j, min_k = np.unravel_index(np.argmin(maes, axis=None), maes.shape)
    min_sigma = sigmas[min_j]
    min_l2reg = l2regs[min_k]
    print('min mae', maes[min_j, min_k], 'for sigma=', min_sigma, 'and l2reg=', min_l2reg)
    return min_sigma, min_l2reg

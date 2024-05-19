import numpy as np
import pickle
import scipy.linalg as linalg
from matplotlib import pyplot as plt


#TODO: cleanup and verify correctness
def DMD(X, svd_rank = None):
    X_1, X_2 = X[ : -1, ...], X[1 :, ...]
    
    # 1) SVD of X_1
    U, s, V_h = linalg.svd(X_1, full_matrices = False)

    zero_thresh = 1e-30
    s[s < zero_thresh] = 0

    #SVD truncation
    if svd_rank is not None and svd_rank < s.shape[0] and svd_rank > 0:
        # U[svd_rank:] = 0
        s[svd_rank:] = 0
        # V_h[:, svd_rank:] = 0
    
    U_h, Sigma, V = linalg.pinv(U), np.diag(s), linalg.pinv(V_h)
    Sigma_inv = np.where(Sigma >= zero_thresh, 1 / (Sigma + 1e-20), 0)

    # 2) get A_tilde, or the estimated koopman matrix
    A_tilde = U_h @ X_2 @ V @ Sigma_inv

    # 3) eigendecomp of A_tilde (W is eigvecs)
    eigvals, W = linalg.eig(A_tilde)
    # print(np.diag(Sigma_inv))

    # 4) reconstruct eigendecomp of A 
    phi = X_2 @ V @ Sigma_inv @ W

    # amplitudes
    amplitudes = linalg.pinv(phi) @ X_1[:, 0]

    # print(amplitudes[0])
    # print(s[0])
    print(phi)


if __name__ == '__main__':
    for task in ['door']: #door hammer relocate
        with open(f'./lifted_traj_{task}.pickle', 'rb') as infile:
            data = pickle.load(infile)

            
            episode = data[0] 
            DMD(episode, 50)

            #for each trajectory in data, run DMD and get modes, compare each by koopman A

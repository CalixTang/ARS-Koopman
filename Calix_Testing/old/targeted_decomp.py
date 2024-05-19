import numpy as np
import pickle
import scipy.linalg as linalg
# from matplotlib import pyplot as plt
task = 'relocate'
with open(f'./lifted_traj_{task}.pickle', 'rb') as infile:
    data = pickle.load(infile)
    first_ep = data[0]

#koopman matrix K
koopman_operator = np.load(f'../hand_dapg/dapg/controller_training/koopman_without_vel/{task}/koopmanMatrix.npy')
eigvals, W = linalg.eig(koopman_operator) #eigvals are sorted from small to big eigval

#amplitudes are useful when doing proper DMD, but we can afford to just eigendecompose entire koopman mat
amplitudes = linalg.pinv(W) @ first_ep[:, 0]


#generate reconstructed matrices from each mode in order of large to small eigenvalue
n = eigvals.shape[0]

# every indiviudal mode
# for i in range(n - 1, -1, -1):
#     A_tilde = W[:, i : i + 1] @ np.diag(eigvals[i : i + 1]) @ linalg.pinv(W[:, i : i + 1])
#     np.save(f'./koopman_mats/{task}_mode_{i}.npy', A_tilde.real)

# #n most dominant modes in increments of 5 up to n
# for i in range(5, n, 5):
#     A_tilde = W[:, -i : ] @ np.diag(eigvals[-i : ]) @ linalg.pinv(W[:, -i : ])
#     np.save(f'./koopman_mats/{task}_top_{i}_modes.npy', A_tilde.real)

# n% most dominant modes in increments of 5 up to 95
# for i in range(5, 100, 5):
#     chosen_modes = np.random.choice(n, int(i / 100 * n), replace=False)
#     A_tilde = W[:, chosen_modes] @ np.diag(eigvals[chosen_modes]) @ linalg.pinv(W[:, chosen_modes])
#     np.save(f'./koopman_mats/{task}_random_{i}_pct_modes.npy', A_tilde.real)

# for i in range(n - 1, -1, -1):
#     A_tilde = np.load(f'./koopman_mats/{task}_mode_{i}.npy')
#     print(A_tilde)

#10 mode chunks
for i in range(10):
    A_tilde = W[:, int(n * i / 10) : int(n * (i + 1) / 10)] @ np.diag(eigvals[int(n * i / 10) : int(n * (i + 1) / 10)]) @  linalg.pinv(W[:, int(n * i / 10) : int(n * (i + 1) / 10)])
    np.save(f'./koopman_mats/{task}_chunk_{i}.npy', A_tilde.real)
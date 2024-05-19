import pickle
import os
import numpy as np
from pydmd import DMD, BOPDMD
from pydmd.plotter import plot_eigs, plot_summary
from pydmd.preprocessing import hankel_preprocessing
import matplotlib.pyplot as plt

svd_rank_map = {'door': 50, 'hammer': 40, 'relocate': 100}

for task in ['door']: #door hammer relocate
    with open(f'./lifted_traj_{task}.pickle', 'rb') as infile:
        data = pickle.load(infile)
        #print(f"data is {data}")

        #koopman matrix K
        # koopman_operator = np.load(f'../hand_dapg/dapg/controller_training/koopman_without_vel/{task}/koopmanMatrix.npy')

        print(data.shape) #should have 200 eps, kodex.shape[1] state dim, and door task timesteps dim
        d = 2
        episode = data[0, ...]
        plt.imshow(episode)
        plt.savefig(f'orig_data{task}.jpg')


        # Build the Optimized DMD model.
        # num_trials=0 gives Optimized DMD, without bagging.
        optdmd = BOPDMD(svd_rank = svd_rank_map[task], num_trials=0, varpro_opts_dict = {"verbose": True, 'maxiter': 30})

        # Wrap the model with the preprocessing routine.
        delay_optdmd = hankel_preprocessing(optdmd, d=d)

        # Fit the model to the noisy data.
        # Note: BOPDMD models need the data X and the times of data collection t for fitting.
        # Hence if we apply time-delay, we must adjust the length of our time vector accordingly.
        delay_t = np.linspace(0, episode.shape[1] - 1, episode.shape[1] - 1, endpoint = False)

        # for ctr in range(data.shape[0]):
        #     delay_optdmd.fit(data[ctr], t=delay_t)
        delay_optdmd.fit(episode, t=delay_t)

        # Plot a summary of the DMD results.
        plot_summary(delay_optdmd, x=data[0], d=d, max_sval_plot = svd_rank_map[task], filename = f'dmd_summary_{task}.jpg')

        # Print computed eigenvalues (frequencies are given by imaginary components).
        # Also plot the resulting data reconstruction.
        print(
            f"Frequencies (imaginary component): {np.round(delay_optdmd.eigs, decimals=3)}"
        )
        plt.title("Reconstructed Data")
        plt.imshow(delay_optdmd.reconstructed_data.real)
        plt.show()
        plt.savefig(f'recon_data{task}.jpg')
        # plt.title("Clean Ground Truth Data")
        # plt.imshow(X.T)
        # plt.show()


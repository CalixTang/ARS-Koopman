"""
This script is based on Yunhai's original script for least-squares solving for a Koopman matrix given data https://github.com/GT-STAR-Lab/KODex/blob/master/Relocation/Koopman_training.py

The math here can be seen as an application of this: https://jxiv.jst.go.jp/index.php/jxiv/preprint/view/602/1858

I've tried to remove unnecessary things as much as possible while keeping the overall code structure the same. Feel free to rewrite this as necessary.
"""

# from cProfile import label
# from glob import escape
# from attr import asdict
# import torch
# import mj_envs
# import click 
import argparse
# import json
import os
import numpy as np
# import gym
# import pickle
from tqdm import tqdm
# from utils.gym_env import GymEnv
# from utils.Observables import *
from utils.Koopman_evaluation import *
# from utils.Controller import *
# from utils.quatmath import quat2euler, euler2quat
# from utils.coord_trans import ori_transform, ori_transform_inverse
from datetime import datetime
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from scipy.linalg import logm
import scipy
# import sys
import os
# import random
import time
import shutil

from ARS.Observables import *

# DESC = '''
# Helper script to visualize demonstrations.\n
# USAGE:\n
#     Visualizes demonstrations on the env\n
#     $ python utils/visualize_demos --env_name relocate-v0\n
# '''

# MAIN =========================================================
def main(demo_file, num_demo, velocity, save_matrix, matrix_file, folder_name):
    num_demo = int(num_demo) # if num_demo != 0 -> we manually define the num of demo for testing the sample efficiency
    Velocity = True if velocity == 'True' else False
    save_matrix = True if save_matrix == 'True' else False  # save the Koopman matrix for Drafted method and trained model for MLP/GNN
    # if demo_file is None:
    #     demos = pickle.load(open('./demonstrations/'+env_name+'_demos.pickle', 'rb'))
    # else:
    #     demos = pickle.load(open(demo_file, 'rb'))
    
    # load demos
    demos = np.load(demo_file, allow_pickle = True)

    if num_demo == 0 or num_demo > len(demos):
        num_demo = len(demos)
   
    error_calc = 'median' # median error or mean error
    unit_train = False
    if len(folder_name) == 0: # we do not explicitly define the folder name for saving the models
        folder_name = '/Relocate/'  # define the folder name for training
        unit_train = True
    if not os.path.exists(matrix_file):  # loading data for training

        #set up training data...
        Training_data = np.load(demo_file)[: num_demo] # len(Training_data) = num_demo
        num_handpos = len(Training_data[0][0]['handpos'])
        num_handvel = len(Training_data[0][0]['handvel'])
        num_objpos = len(Training_data[0][0]['objpos']) + len(Training_data[0][0]['objorient'])
        num_objvel = len(Training_data[0][0]['objvel'])
        dt = 1  # simulation time for each step, assuming it should be 1
        num_obj = num_objpos + num_objvel
        if Velocity: # hand velocities are also used as the original states
            num_hand = num_handpos + num_handvel
        else:
            num_hand = num_handpos
    
    '''
    Train the koopman dynamics from demo data
    '''

    # record the training errors
    # Drafted: error over num of used trajectories 
    # MLP: error over num of training iterations
    # GNN: error over num of training iterations
    # Start the training process for each mode
    if unit_train:
        parent_dir = './Results/' + koopmanoption + folder_name
        current_time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        os.mkdir(os.path.join(parent_dir, current_time))
        results_record = open('./Results/' + koopmanoption + folder_name + current_time + '/results.txt', 'w+')
    else:
        if not os.path.exists(os.path.join(folder_name, "koopman")):
            os.mkdir(os.path.join(folder_name, "koopman"))
        else:
            shutil.rmtree(os.path.join(folder_name, "koopman"))   # Removes all the subdirectories!
            os.mkdir(os.path.join(folder_name, "koopman"))
        results_record = open(folder_name + "koopman" + '/results.txt', 'w+')
    start_time = time.time()

    #we only use "Drafted" observables for now (hardcoded lifting functions)
    # if koopmanoption == "Drafted":

    Koopman = DraftedObservable(num_hand, num_obj)
    num_obs = Koopman.compute_observable(num_hand, num_obj)
    print("number of observables:", num_obs)
    A = np.zeros((num_obs, num_obs))  
    G = np.zeros((num_obs, num_obs))
    ## loop to collect data
    print("Drafted koopman training starts!\n")
    for k in tqdm(range(len(Training_data))):
        if Velocity:
            hand_OriState = np.append(Training_data[k][0]['handpos'], Training_data[k][0]['handvel'])
        else:
            hand_OriState = Training_data[k][0]['handpos']
        obj_OriState = np.append(Training_data[k][0]['objpos'], np.append(Training_data[k][0]['objorient'], Training_data[k][0]['objvel']))
        z_t = Koopman.z(hand_OriState, obj_OriState)  # initial states in lifted space
        for t in range(len(Training_data[k]) - 1):
            if Velocity:
                hand_OriState = np.append(Training_data[k][t + 1]['handpos'], Training_data[k][t + 1]['handvel'])
            else:
                hand_OriState = Training_data[k][t + 1]['handpos']
            obj_OriState = np.append(Training_data[k][t + 1]['objpos'], np.append(Training_data[k][t + 1]['objorient'], Training_data[k][t + 1]['objvel']))
            z_t_1 = Koopman.z(hand_OriState, obj_OriState) # states in lifted space at next time step
            # A and G are cumulated through all the demo data
            A += np.outer(z_t_1, z_t)
            G += np.outer(z_t, z_t)
            z_t = z_t_1
    M = len(Training_data) * (len(Training_data[0]) - 1)
    A /= M
    G /= M
    koopman_operator = np.dot(A, scipy.linalg.pinv(G)) # do not use np.linalg.pinv, it may include all large singular values
    # cont_koopman_operator = logm(koopman_operator) / dt
    cont_koopman_operator = koopman_operator
    results_record.write("Training time is: %f seconds.\n"%(time.time() - start_time))
    # generate another matrix with similar matrix magnitude to verify the correctness of the learnt koopman matrix
    # we want to see that each element in the matrix does matter
    koopman_mean = np.mean(cont_koopman_operator)
    print("Koopman mean:%f"%(koopman_mean))
    koopman_std = np.std(cont_koopman_operator)
    print("Koopman std:%f"%(koopman_std))
    Test_matrix = np.random.normal(loc = koopman_mean, scale = koopman_std, size = cont_koopman_operator.shape)
    print("Fake matrix mean:%f"%(np.mean(Test_matrix)))
    print("Fake matrix std:%f"%(np.std(Test_matrix)))
    print("Drafted koopman training ends!\n")
    # print("The drafted koopman matrix is: ", cont_koopman_operator)
    # print("The drafted koopman matrix shape is: ", koopman_operator.shape)
            # save the trained koopman matrix

    #always save the matrix
    if unit_train:
        np.save('./Results/Drafted' + folder_name + current_time + '/koopmanMatrix.npy', cont_koopman_operator)
    else:
        np.save(folder_name + "koopman" + '/koopmanMatrix.npy', cont_koopman_operator)
        print("Koopman matrix is saved!\n")
    
    print("Koopman final testing starts!\n")
    ErrorInLifted, ErrorInOriginal = koopman_evaluation(Koopman, cont_koopman_operator, Training_data, Velocity, num_hand, num_obj)
    Fake_ErrorInLifted, Fake_ErrorInOriginal = koopman_evaluation(Koopman, Test_matrix, Training_data, Velocity, num_hand, num_obj)
    print("Koopman final testing ends!\n")
    if error_calc == 'median': # compute the median 
        print("The final test accuracy in lifted space is: %f, and the accuracy in original space is: %f."%(np.median(ErrorInLifted), np.median(ErrorInOriginal)))
        print("The fake test accuracy in lifted space is: %f, and the fake accuracy in original space is: %f."%(np.median(Fake_ErrorInLifted), np.median(Fake_ErrorInOriginal)))
        results_record.write("The final test accuracy in lifted space is: %f, and the accuracy in original space is: %f.\n"%(np.median(ErrorInLifted), np.median(ErrorInOriginal)))
        results_record.write("The fake test accuracy in lifted space is: %f, and the fake accuracy in original space is: %f.\n"%(np.median(Fake_ErrorInLifted), np.median(Fake_ErrorInOriginal)))
    else:
        print("The final test accuracy in lifted space is: %f, and the accuracy in original space is: %f."%(np.mean(ErrorInLifted), np.mean(ErrorInOriginal)))
        print("The fake test accuracy in lifted space is: %f, and the fake accuracy in original space is: %f."%(np.mean(Fake_ErrorInLifted), np.mean(Fake_ErrorInOriginal)))
        results_record.write("The final test accuracy in lifted space is: %f, and the accuracy in original space is: %f.\n"%(np.mean(ErrorInLifted), np.mean(ErrorInOriginal)))
        results_record.write("The fake test accuracy in lifted space is: %f, and the fake accuracy in original space is: %f.\n"%(np.mean(Fake_ErrorInLifted), np.mean(Fake_ErrorInOriginal)))
    results_record.write("The number of demo used for experiment is: %d.\n"%(num_demo))
    results_record.flush()
    results_record.close()


# This function is taken from https://github.com/GT-STAR-Lab/KODex/blob/b058159f59baeff37437672e49f5d73406d4a5a6/Relocation/utils/Koopman_evaluation.py#L14. It is included in an import in the original koopman_train file, but I copy it into this file since I only use it locally.
# Evaluates the koopman matrix based on its fit with test data. 
def koopman_evaluation(koopman_object, koopman_matrix, eval_episodes):
    '''
    Input: koopman_object - Koopman object (Drafted, MLP, GNN) for observable lifting
           koopman_matrix - Learned koopman matrix
           eval_episodes - 3d np array of (episode_dim, timestep_dim, obs_dim)
    '''
    (n_episodes, timesteps_per_episode, obs_dim) = eval_episodes.shape

    ErrorInLifted = np.zeros(koopman_object.compute_observables_from_self())
    ErrorInOriginal = np.zeros(obs_dim)

    for k in tqdm(range(n_episodes)):
        # if Velocity:
        #     hand_OriState = np.append(eval_episodes[k][0]['handpos'], eval_episodes[k][0]['handvel'])
        # else:
        #     hand_OriState = eval_episodes[k][0]['handpos']

        #get initial state and initial lifted state
        x_t = eval_episodes[k, 0, :]
        z_t = koopman_object.z(x_t)

        for t in range(len(eval_episodes[k]) - 1):

            #get ground truth next state and next lifted state
            x_t_1 = eval_episodes[k, t + 1, :]
            z_t_1 = koopman_object.z(x_t_1)

            #estimate next state and next lifted state using k matrix
            z_t_1_computed = np.dot(koopman_matrix, z_t)
            x_t_1_computed = z_t_1_computed[ : obs_dim] #MAJOR ASSUMPTION: koopman object's z() leaves the original observations in the front of the array
            
            ErrorInLifted += np.abs(z_t_1 - z_t_1_computed)  # if using np.square, we will get weird results. <- Yunhai's design choice that i will leave alone
            ErrorInOriginal += np.abs(x_t_1 - x_t_1_computed)
            z_t = z_t_1

    M = n_episodes * (timesteps_per_episode - 1)
    ErrorInLifted /= M
    ErrorInOriginal /= M
    return ErrorInLifted, ErrorInOriginal

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo_file', type=str, required = True, help='demo file to load')
    parser.add_argument('--num_demo', type=int, help='define the number of demo', default=0) 
    # parser.add_argument('--koopmanoption', type=str, help='Indicate the koopman choice (Drafted, MLP, GNN)', default='Drafted')   # We only use Drafted in my project
    parser.add_argument('--velocity', type=str, help='If using hand velocity', default=None)
    parser.add_argument('--save_matrix', type=str, help='If save the koopman matrix after training', default=None)
    parser.add_argument('--matrix_file', type=str, help='If loading the saved matrix/model for test', default='')
    parser.add_argument('--config', type=str, help='load the network info for MLP and GNN', default='')
    parser.add_argument('--control', type=str, help='apply with a controller', default='')
    parser.add_argument('--error_type', type=str, help='define how to calculate the errors', default='demo') # two options: demo; goal
    parser.add_argument('--visualize', type=str, help='define whether or not to visualze the manipulation results', default='') # two options: demo; goal
    parser.add_argument('--unseen_test', type=str, help='define if we generate unseen data for testing the learned dynamics model', default='') # generate new testing data
    parser.add_argument('--rl_policy', type=str, help='define the file location of the well-trained policy', default='') 
    parser.add_argument('--folder_name', type=str, help='define the location to put trained models', default='') 
    args = parser.parse_args()

    main()


# class Observable(object):
#     """
#         Base class for an observable. Should lift a state vector to a lifted state vector.
#     """
#     def __init__(self, num_envStates):
#         self.num_states = num_envStates

#     """
#     Implementation of lifting function
#     """
#     def z(self, envState):
#         raise NotImplementedError
    
#     """
#     Compute the size of lifted state given size of state 
#     """
#     def compute_observable(num_states):
#         raise NotImplementedError
    
#     """
#     Compute the size of lifted state given size of state for this instance.
#     """
#     def compute_observables_from_self(self):
#         raise NotImplementedError

# """
# A larger observable because we might need it
# """
# class LargeManipulationObservable(Observable):
#     def __init__(self, num_envStates):
#         super().__init__(num_envStates)

#     def z(self, envState):
#         """
#         Lifts the environment state from state space to full "observable space' (Koopman). g(x) = z.
#         Inputs: environment states
#         Outputs: state in lifted space
#         """
#         obs = np.zeros(self.compute_observables_from_self())
#         index = 0

#         #x[i]
#         obs[index : index + self.num_states] = envState[:]
#         index += self.num_states

#         #x[i]^2
#         obs[index : index + self.num_states] = envState ** 2
#         index += self.num_states  

#         #sin(x)
#         obs[index : index + self.num_states] = np.sin(envState)
#         index += self.num_states

#         #cos(x)
#         obs[index : index + self.num_states] = np.cos(envState)
#         index += self.num_states

#         #x[i] x[j] w/o repetition
#         obs[index : index + (self.num_states * (self.num_states - 1)) // 2] = np.outer(envState, envState)[np.triu_indices(self.num_states - 1, k=1)]
#         index += (self.num_states * (self.num_states - 1)) // 2  

#         # x[i]^2 x[j] w/ repetition - I removed the x[i]^3 here b/c this block includes x[i]^3
#         obs[index : index + (self.num_states ** 2)] = np.outer(envState ** 2, envState).flatten()
#         index += self.num_states ** 2


#     def compute_observable(num_states):
#         return 4 * num_states + (num_states * (num_states - 1)) // 2 + num_states ** 2
    
#     def compute_observables_from_self(self):
#         return LargeManipulationObservable.compute_observable(self.num_states)
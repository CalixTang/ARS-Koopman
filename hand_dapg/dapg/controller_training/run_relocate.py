"""
This is a job script for controller learning for KODex 1.0
"""

from re import A
from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.algos.npg_cg import NPG
from mjrl.utils.train_agent import train_agent
from mjrl.samplers.core import sample_paths
from mjrl.KODex_utils.Observables import *
from mjrl.KODex_utils.quatmath import quat2euler, euler2quat
from mjrl.KODex_utils.coord_trans import ori_transform, ori_transform_inverse
from mjrl.KODex_utils.Controller import *
import sys
import os
import json
import mjrl.envs
import mj_envs   # read the env files (task files)
import time as timer
import pickle
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.linalg as linalg

# using the recommendated fig params from https://github.com/jbmouret/matplotlib_for_papers#pylab-vs-matplotlib
fig_params = {
'axes.labelsize': 10,
'axes.titlesize':15,
'font.size': 10,
'legend.fontsize': 10,
'xtick.labelsize': 10,
'ytick.labelsize': 10,
'text.usetex': False,
'figure.figsize': [5, 4.5]
}
mpl.rcParams.update(fig_params)

def demo_playback(demo_paths, num_demo, task_id):
    Training_data = []
    print("Begin loading demo data!")
    # sample_index = np.random.choice(len(demo_paths), num_demo, replace=False)  # Random data is used
    sample_index = range(num_demo)
    for t in sample_index:  # only read the initial conditions
        path = demo_paths[t]
        state_dict = {}
        if task_id == 'pen':
            observations = path['observations']  
            handVelocity = path['handVelocity']  
            obs = observations[0] # indeed this one is defined in the world frame(fixed on the table) (for object position and object orientations)
            state_dict['init_states'] = path['init_state_dict']
            state_dict['handpos'] = obs[:24]
            state_dict['handvel'] = handVelocity[0]
            state_dict['objpos'] = obs[24:27] # in the world frame(on the table)
            state_dict['objvel'] = obs[27:33]
            state_dict['desired_ori'] = obs[36:39] # desired orientation (an unit vector in the world frame)
            state_dict['objorient'] = obs[33:36] # initial orientation (an unit vector in the world frame)
        elif task_id == 'relocate':
            observations = path['observations']  
            observations_visualize = path['observations_visualization']
            handVelocity = path['handVelocity'] 
            obs = observations[0] 
            obs_visual = observations_visualize[0]
            state_dict['init_states'] = path['init_state_dict']
            state_dict['handpos'] = obs[:30]
            state_dict['handvel'] = handVelocity[0][:30]
            objpos = obs[39:42] # in the world frame(on the table)
            state_dict['desired_pos'] = obs[45:48] 
            state_dict['objpos'] = objpos - obs[45:48] # converged object position
            state_dict['objorient'] = obs_visual[33:36]
            state_dict['objvel'] = handVelocity[0][30:]
        elif task_id == 'door':
            observations = path['observations']  
            observations_visualize = path['observations_visualization']
            obs = observations[0] 
            obs_visual = observations_visualize[0]
            state_dict['init_states'] = path['init_state_dict']
            state_dict['handpos'] = obs_visual[:28]
            state_dict['handvel'] = obs_visual[30:58]
            state_dict['objpos'] = obs[32:35]  # handle position
            state_dict['objvel'] = obs_visual[58:59]  # door hinge
            state_dict['handle_init'] = path['init_state_dict']['door_body_pos']
        elif task_id == 'hammer':
            observations = path['observations'] 
            handVelocity = path['handVelocity'] 
            obs = observations[0]
            allvel = handVelocity[0]
            state_dict['init_states'] = path['init_state_dict']
            state_dict['handpos'] = obs[:26]
            state_dict['handvel'] = allvel[:26]
            state_dict['objpos'] = obs[49:52] + obs[42:45] 
            state_dict['objorient'] = obs[39:42]
            state_dict['objvel'] = obs[27:33]
            state_dict['nail_goal'] = obs[46:49]
        Training_data.append(state_dict)
    print("Finish loading demo data!")
    return Training_data


#run
if __name__ == '__main__':
    # ===============================================================================
    # Get command line arguments
    # ===============================================================================
    parser = argparse.ArgumentParser(description='Policy gradient algorithms with demonstration data.')
    parser.add_argument('--config', type=str, required=True, help='path to config file with exp params')
    parser.add_argument('--policy', type=str, default = None, required=False, help='absolute path of the policy file')
    parser.add_argument('--eval_data', type=str, required=True, help='absolute path to evaluation data')
    parser.add_argument('--visualize', type=str, required=True, help='determine if visualizing the policy or not')
    parser.add_argument('--save_fig', type=str, required=True, help='determine if saving all generated figures')
    parser.add_argument('--record_video', required = False, action = 'store_true', help='determine if only recording the policy rollout')
    parser.add_argument('--save_traj_path', type = str, required = False)
    parser.add_argument('--object', type = str, required = False)
    parser.add_argument('--koopman_path', type=str, required=False, default = None, help = 'path to koopman matrix')
    parser.add_argument('--dynamic_mode_path', type=str, required = False, default = None, help = 'path to dynamic mode eigenvectors')
    parser.add_argument('--video_path', required = False, default = './Videos/CIMER.mp4')
    parser.add_argument('--num_episodes', type = int, required= False, default = -1)

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        job_data = eval(f.read())
    assert 'algorithm' in job_data.keys()
    visualize = False
    if args.visualize == "True":
        visualize = True
    Save_fig = False
    if args.save_fig == "True":
        Save_fig = True
    Only_record_video = False
    if args.record_video:
        Only_record_video = True
    assert any([job_data['algorithm'] == a for a in ['NPG', 'TRPO', 'PPO']])  # start from natural policy gradient for training
    assert os.path.exists(os.getcwd() + job_data['matrix_file'] + job_data['env'].split('-')[0] + '/koopmanMatrix.npy')  # loading KODex reference dynamics
    KODex = None
    if args.koopman_path is not None:
        print(f"Loading KODex from provided path {args.koopman_path}")
        KODex = np.load(args.koopman_path)
    else:
        if args.dynamic_mode_path is not None:
            print("Reconstructing KODex from dynamic modes with uniform weighting")
            eigvecs = np.load(args.dynamic_mode_path)
            default_weights = np.diag(1 - 2 * np.random.rand(eigvecs.shape[1]))
            print(default_weights)
            KODex = eigvecs @ default_weights @ linalg.pinv(eigvecs)
        else:
            print("Loading KODex from job file arguments")
            KODex = np.load(os.getcwd() + job_data['matrix_file'] + job_data['env'].split('-')[0] + '/koopmanMatrix.npy')

    # ===============================================================================
    # Set up the controller parameter
    # ===============================================================================
    # This set is used for control frequency: 500HZ (for all DoF)
    PID_P = 10
    PID_D = 0.005  
    Simple_PID = PID(PID_P, 0.0, PID_D)
    # ===============================================================================
    # Task specification
    # ===============================================================================
    task_id = 'relocate'
    Obj_sets = ['banana', 'cracker_box', 'cube', 'cylinder', 'foam_brick', 'gelatin_box', 'large_clamp', 'master_chef_can', 'mug', 'mustard_bottle', 'potted_meat_can', 'power_drill', 'pudding_box', 'small_ball', 'sugar_box', 'tomato_soup_can', 'tuna_fish_can']
    try: 
        if job_data['object'] not in Obj_sets:
            job_data['object'] = ''  # default setting
    except:
        job_data['object'] = '' # default setting

    if args.object is not None:
        job_data['object'] = args.object
    num_robot_s = 30
    num_object_s = 12
    task_horizon = 100
    # ===============================================================================
    # Train Loop
    # ===============================================================================
    if job_data['control_mode'] not in ['Torque', 'PID']:
        print('Unknown action space! Please check the parameter control_mode in the job script.')
        sys.exit()
    # Visualization

    policy = None
    if args.policy is None:
        special_policies = ['cracker_box', 'cube', 'cylinder', 'master_chef_can', 'pudding_box', 'small_ball']
        if job_data['object'] in special_policies:
            policy = pickle.load(open(f'Samples/Relocate/CIMER/{job_data["object"]}.pickle', 'rb'))
        else:
            policy = pickle.load(open('Samples/Relocate/CIMER/best_eval_sr_policy.pickle', 'rb'))
    else:
        policy = pickle.load(open(args.policy, 'rb')) 
    
    e = GymEnv(job_data['env'], job_data['control_mode'], job_data['object'])  # an unified env wrapper for all kind of envs
    
    Koopman_obser = DraftedObservable(num_robot_s, num_object_s)
    demos = pickle.load(open(args.eval_data, 'rb'))

    num_episodes = len(demos) if args.num_episodes == -1 else args.num_episodes

    Eval_data = demo_playback(demos, len(demos), task_id)
    coeffcients = dict()
    coeffcients['task_ratio'] = job_data['task_ratio']
    coeffcients['tracking_ratio'] = job_data['tracking_ratio']
    coeffcients['hand_track'] = job_data['hand_track']
    coeffcients['object_track'] = job_data['object_track']
    coeffcients['ADD_BONUS_REWARDS'] = 1 # when evaluating the policy, it is always set to be enabled
    coeffcients['ADD_BONUS_PENALTY'] = 1
    gamma = 1.
    print("gamma:", gamma)
    try:
        policy.freeze_base
    except:
        policy.freeze_base = False        
    try:
        policy.include_Rots
    except:
        policy.include_Rots = False   
    print(policy.m)
    # plug into a NN-based controller for Door task and test its performance, and also, tried to be compatible with the 24 DoFs training case.
    # load a door opening controller, which is important to 
    if not Only_record_video:
        z_mat = e.record_trajectories(Eval_data, Simple_PID, coeffcients, Koopman_obser, KODex, task_horizon, job_data['future_s'], job_data['history_s'], policy, num_episodes=len(demos), gamma = gamma, obj_dynamics = job_data['obj_dynamics'], visual = visualize)  # noise-free actions
        print(z_mat.shape)
        # score[0][0] -> mean sum rewards for rollout (task-specific rewards)
        # because we are now in evaluation mode, the reward/score here is the sum of gamma_discounted rewards at t=0 (the sum rewards at other states at t=1,..,T are ignored) 
        # In other words, this can be seen as the reward in the domain of trajectory.
        with open(args.save_traj_path, 'wb+') as outfile:
            np.save(outfile, z_mat)

        if Save_fig:
            root_dir = os.getcwd() + "/" + args.policy[:args.policy.find('.')]
            if not os.path.exists(root_dir):
                os.mkdir(root_dir)
            force_compare_index, MA_PD_tips_force, PD_tips_force, Joint_adaptations = force_values
            all_index = [i for i in range(len(Eval_data))]
            force_compare_index = all_index # compare the performance across all samples
            Joint_changes = np.zeros([num_robot_s, task_horizon - 1])
            Joints_name = {"Base":["A_ARTx", "A_ARTy", "A_ARTz", "A_ARRx", "A_ARRy", "A_ARRz"], "Wrist":["A_WRJ1", "A_WRJ0"], "Forefinger":["A_FFJ3", "A_FFJ2", "A_FFJ1", "A_FFJ0"], "MiddleFinger":["A_MFJ3", "A_MFJ2", "A_MFJ1", "A_MFJ0"], "RingFinger":["A_RFJ3", "A_RFJ2", "A_RFJ1", "A_RFJ0"], "LittleFinger":["A_LFJ4", "A_LFJ3", "A_LFJ2", "A_LFJ1", "A_LFJ0"], "Thumb":["A_THJ4", "A_THJ3", "A_THJ2", "A_THJ1", "A_THJ0"]}
            for i in range(len(force_compare_index)):
                for j in range(len(Joint_adaptations[force_compare_index[i]])):
                    Joint_changes[:, j] += Joint_adaptations[force_compare_index[i]][j] / len(force_compare_index)
            finger_index = 0
            for i in range(len(Joints_name.keys())):
                body_type = list(Joints_name.keys())[i]
                if body_type != "Wrist":  # Wrist only has two parts, so one row is enough
                    plt.figure(i)
                    fig, ax = plt.subplots(2, 3)
                    for j in range(len(Joints_name[body_type])):
                        ax[j // 3 , j % 3].plot(Joint_changes[finger_index], linewidth=1, color='#B22400')
                        ax[j // 3 , j % 3].vlines(22, 1.1 * min(Joint_changes[finger_index]), 1.1 * max(Joint_changes[finger_index]), linestyles='dotted', colors='k')
                        ax[j // 3 , j % 3].set_ylim([1.1 * min(Joint_changes[finger_index]), 1.1 * max(Joint_changes[finger_index])])
                        ax[j // 3 , j % 3].set(title=Joints_name[body_type][j])
                        finger_index += 1
                        fig.legend(['Differences of joint targets (%s) made by MA'%(body_type)], loc='lower left')
                        fig.supxlabel('Time step')
                        fig.supylabel('Differences of joint targets')
                else:
                    plt.figure(i)
                    fig, ax = plt.subplots(1, 2)
                    for j in range(2):
                        ax[j % 2].plot(Joint_changes[finger_index], linewidth=1, color='#B22400')
                        ax[j % 2].vlines(22, 1.1 * min(Joint_changes[finger_index]), 1.1 * max(Joint_changes[finger_index]), linestyles='dotted', colors='k')
                        ax[j % 2].set_ylim([1.1 * min(Joint_changes[finger_index]), 1.1 * max(Joint_changes[finger_index])])
                        ax[j % 2].set(title=Joints_name[body_type][j])
                        finger_index += 1
                        fig.legend(['Differences of joint targets (%s) made by MA'%(body_type)], loc='lower left')
                        fig.supxlabel('Time step')
                        fig.supylabel('Differences of joint targets')
                plt.tight_layout()
                plt.savefig(root_dir + '/%s.png'%(body_type))
                plt.close()
            # all_index = [i for i in range(len(Eval_data))]
            # force_compare_index = all_index # compare the performance across all samples
            finger_index = ['ff', 'mf', 'rf', 'lf', 'th', 'sum']
            finger_index_vis = ['Forefinger', 'Middlefinger', 'Ringfinger', 'Littlefinger', 'Thumb', 'Sum']
            MA_PD_total_force = {'ff': np.zeros(task_horizon - 1), 'mf': np.zeros(task_horizon - 1), 'rf': np.zeros(task_horizon - 1), 'lf': np.zeros(task_horizon - 1), 'th': np.zeros(task_horizon - 1), 'sum': np.zeros(task_horizon - 1)}
            PD_total_force = {'ff': np.zeros(task_horizon - 1), 'mf': np.zeros(task_horizon - 1), 'rf': np.zeros(task_horizon - 1), 'lf': np.zeros(task_horizon - 1), 'th': np.zeros(task_horizon - 1), 'sum': np.zeros(task_horizon - 1)}
            MA_PD_PD_ratio = {'ff': np.zeros(task_horizon - 1), 'mf': np.zeros(task_horizon - 1), 'rf': np.zeros(task_horizon - 1), 'lf': np.zeros(task_horizon - 1), 'th': np.zeros(task_horizon - 1), 'sum': np.zeros(task_horizon - 1)}
            # additional visualization codes
            ours_all_force = {'ff': np.zeros([task_horizon - 1, len(force_compare_index)]), 'mf': np.zeros([task_horizon - 1, len(force_compare_index)]), 'rf': np.zeros([task_horizon - 1, len(force_compare_index)]), 'lf': np.zeros([task_horizon - 1, len(force_compare_index)]), 'th': np.zeros([task_horizon - 1, len(force_compare_index)]), 'sum': np.zeros([task_horizon - 1, len(force_compare_index)])}
            KODex_PD_all_force = {'ff': np.zeros([task_horizon - 1, len(force_compare_index)]), 'mf': np.zeros([task_horizon - 1, len(force_compare_index)]), 'rf': np.zeros([task_horizon - 1, len(force_compare_index)]), 'lf': np.zeros([task_horizon - 1, len(force_compare_index)]), 'th': np.zeros([task_horizon - 1, len(force_compare_index)]), 'sum': np.zeros([task_horizon - 1, len(force_compare_index)])}
            ours_KODex_PD_all_force = {'ff': np.zeros([task_horizon - 1, len(force_compare_index)]), 'mf': np.zeros([task_horizon - 1, len(force_compare_index)]), 'rf': np.zeros([task_horizon - 1, len(force_compare_index)]), 'lf': np.zeros([task_horizon - 1, len(force_compare_index)]), 'th': np.zeros([task_horizon - 1, len(force_compare_index)]), 'sum': np.zeros([task_horizon - 1, len(force_compare_index)])}
            # additional visualization codes
            for i in range(len(force_compare_index)):
                plt.figure(i)
                fig, ax = plt.subplots(2, 3)
                for row in range(2):
                    for col in range(3):
                        tmp_length = len(MA_PD_tips_force[finger_index[3*row+col]][force_compare_index[i]])  # -> a list through time horizon
                        MA_PD_minus_PD = list()
                        index_ = 0
                        for item1, item2 in zip(MA_PD_tips_force[finger_index[3*row+col]][force_compare_index[i]], PD_tips_force[finger_index[3*row+col]][force_compare_index[i]][:tmp_length]):
                            MA_PD_total_force[finger_index[3*row+col]][index_] += item1 / len(force_compare_index)
                            PD_total_force[finger_index[3*row+col]][index_] += item2 / len(force_compare_index)
                            # additional visualization codes
                            ours_all_force[finger_index[3*row+col]][index_, i] += item1 
                            KODex_PD_all_force[finger_index[3*row+col]][index_, i] += item2 
                            if item1 / (item2 + 1e-8) > 5:
                                ours_KODex_PD_all_force[finger_index[3*row+col]][index_, i] = 5  
                            else:
                                ours_KODex_PD_all_force[finger_index[3*row+col]][index_, i] = item1 / (item2 + 1e-8)
                            # additional visualization codes
                            MA_PD_minus_PD.append(item1 - item2)
                            index_ += 1
                        ax[row, col].plot(MA_PD_minus_PD, linewidth=1, color='#B22400')
                        # ax[row, col].plot(PD_tips_force[finger_index[3*row+col]][force_compare_index[i]], linewidth=1, color='#F22BB2')
                        ax[row, col].vlines(22, 1.1 * min(MA_PD_minus_PD), 1.1 * max(MA_PD_minus_PD), linestyles='dotted', colors='k')
                        ax[row, col].set_ylim([1.1 * min(MA_PD_minus_PD), 1.1 * max(MA_PD_minus_PD)])
                        ax[row, col].set(title=finger_index[3*row+col])
                        # ax[row, col].legend()
                fig.legend(['MA_PD - PD (larger the value, fingertips are closer to the object)'], loc='lower left')
                fig.supxlabel('Time step')
                fig.supylabel('Difference of touch sensor feedback')
                plt.tight_layout()
                plt.savefig(root_dir + '/touch_sensor_' + str(force_compare_index[i]) + '.png')
                plt.close()

            for finger_ in finger_index:
                for i in range(task_horizon - 1):
                    if MA_PD_total_force[finger_][i] / (PD_total_force[finger_][i] + 1e-8) > 5:
                        MA_PD_PD_ratio[finger_][i] = 5
                    else:
                        MA_PD_PD_ratio[finger_][i] = MA_PD_total_force[finger_][i] / (PD_total_force[finger_][i] + 1e-8)

            # additional visualization codes
            plt.figure(2) # plot values 
            fig, ax = plt.subplots(2, 3)
            for row in range(2):
                for col in range(3):
                    x_simu = np.arange(0, ours_all_force[finger_index[3*row+col]].shape[0])
                    low_ours, mid_ours, high_ours = np.percentile(ours_all_force[finger_index[3*row+col]], [25, 50, 75], axis=1)
                    low_kodex_pd, mid_kodex_pd, high_kodex_pd = np.percentile(KODex_PD_all_force[finger_index[3*row+col]], [25, 50, 75], axis=1)
                    mean_ours, std_ours = np.mean(ours_all_force[finger_index[3*row+col]], axis = 1), np.std(ours_all_force[finger_index[3*row+col]], axis=1)
                    mean_kodex_pd, std_kodex_pd = np.mean(KODex_PD_all_force[finger_index[3*row+col]], axis = 1), np.std(KODex_PD_all_force[finger_index[3*row+col]], axis=1)
                    ax[row, col].plot(x_simu, mean_ours, linewidth = 2, label = 'CIMER', color='purple')
                    ax[row, col].fill_between(x_simu, mean_ours - std_ours, mean_ours + std_ours, alpha = 0.15, linewidth = 0, color='purple')
                    ax[row, col].plot(x_simu, mean_kodex_pd, linewidth = 2, label = 'Imitator + PD', color='b')
                    ax[row, col].fill_between(x_simu, mean_kodex_pd - std_kodex_pd, mean_kodex_pd + std_kodex_pd, alpha = 0.15, linewidth = 0, color='b')
                    ax[row, col].set(title=finger_index_vis[3*row+col])
                    ax[row, col].grid()
                    ax[row, col].tick_params(axis='both', labelsize=16)
            # fig.suptitle("Changes of Grasping Force", fontsize=22)
            fig.supxlabel('Time Step', fontsize=20)
            fig.supylabel('Grasping Force', fontsize=20)
            # legend = fig.legend(fontsize=10, loc='lower left')
            # legend = ax[row, col].legend()
                    # ax[row, col].vlines(22, 0, 1.1 * max(MA_PD_PD_ratio[finger_index[3*row+col]]), linestyles='dotted', colors='k')
            # fig.legend(['Ratio of MA_PD over PD (max: 5)'], loc='lower left')
            plt.tight_layout()
            plt.savefig(root_dir + '/values.png')
    else:  # Only_record_video
        e.record_relocate(Eval_data, Simple_PID, coeffcients, Koopman_obser, KODex, task_horizon, job_data['future_s'], job_data['history_s'], num_episodes=num_episodes, gamma = gamma, obj_dynamics = job_data['obj_dynamics'], visual = visualize, vid_path = args.video_path)
    

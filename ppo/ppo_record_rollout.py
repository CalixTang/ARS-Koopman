'''
this PPO2 implementation is taken from https://github.com/ericyangyu/PPO-for-Beginners/blob/master/part4/ppo_for_beginners/ppo_optimized.py
'''


"""
    The file contains the PPO class to train with.
    NOTE: All "ALG STEP"s are following the numbers from the original PPO pseudocode.
            It can be found here: https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg
"""

# import gym
import gymnasium as gym #allow importing robosuite
import time

import numpy as np
import time
# import torch
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from tqdm import tqdm
import imageio

import os

#check if the dependency on logz (in ARS package) is what we want
from ARS import logz
from ARS.graph_results import graph_training_and_eval_rewards

from ppo_policies import TruncatedKoopmanNetworkPolicy, MinKoopmanNetworkPolicy, NNPolicy, get_policy
from torch_observables import LocomotionObservableTorch, LargeManipulationObservableTorch
from koopmanutils.env_utils import handle_extra_params, get_state_pos_and_vel_idx, instantiate_gym_env

#modified version of run_policy.py and record_relocate from mjrl.utils.gym_env
def record_rollouts(task_id='HalfCheetah-v2',
                policy_params = None,
                actor = None,
                logdir=None,
                num_rollouts=50,
                rollout_length = 500,
                shift = 0.):
    
    #can't use vectorized env :/
    env = instantiate_gym_env(task_id, policy_params)

    save_path = os.path.join(logdir, f"{task_id}_eval_{num_rollouts}_rollouts.mp4")
    vid_writer = imageio.get_writer(save_path, mode = 'I', fps = 60)
    
    # env.viewer_setup()
    
    total_reward = 0.
    steps = 0

    ep_rewards = np.zeros(num_rollouts)

    for i in tqdm(range(num_rollouts)):

        obs, _ = env.reset() #for v4 envs
        episode_reward = 0
        for t in range(rollout_length):
            #generate torque action
            action = actor(obs).detach().numpy()
            reward = 0

            #TODO: verify if we need to be using koopman op on the next_o or the actually observed env state
            #(strict koopman trajectory that we follow vs doing a simple "koopman-ish" update on observed state as is implemented here)
            obs, reward, terminated, done, info = env.step(action)   #for v4 env

            # res = env.render(mode = 'rgb_array')
            res = env.render() 
            # res = env.render(mode = 'rgb_array', width = vid_res[0], height = vid_res[1])
            vid_writer.append_data(res)

            episode_reward += (reward - shift)
            if np.all(terminated) or np.all(done):
                break
        
        ep_rewards[i] = episode_reward
                
    vid_writer.close()
    return ep_rewards


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser()

    parser.add_argument('--logdir', type = str, required = True, default = 'data')
    parser.add_argument('--actor_weight_file', type=str, default = 'ppo_actor.pth') #this should be contained in logdir
    parser.add_argument('--num_rollouts', type = int, default = 20)
    parser.add_argument('--vid_res', default = [720, 640])

    args = parser.parse_args()

    params = json.load(open(os.path.join(args.logdir, 'params.json'), 'r'))

    env = gym.make(params['task_id'])

    obs_dim = 0
    if isinstance(env.observation_space, gym.spaces.dict.Dict):
        obs_dim = sum([v.shape[0] for k, v in env.observation_space.items()])
    else:
        obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    state_pos_idx, state_vel_idx = get_state_pos_and_vel_idx(params['task_id'])

    actor = get_policy(params['policy_type'], obs_dim, act_dim, params['lifting_function'], state_pos_idx, state_vel_idx, params['PDctrl_P'], params['PDctrl_D'])
    actor_weight_path = os.path.join(args.logdir, args.actor_weight_file)
    actor.load_state_dict(torch.load(actor_weight_path))
    actor.eval()

    policy_params={'task_id' : params.get('task_id', 'FetchPush-v2'),
                   'rollout_length' : params.get('rollout_length', 50),
                   'lifting_function': params.get('lifting_function', 'locomotion'),
                   'obs_pos_idx': state_pos_idx,
                   'obs_vel_idx': state_vel_idx,
                   'reward_type': params.get('reward_type', 'dense'),
                   'render_mode': 'rgb_array', #this is specific to recording rollouts
                   'seed': params.get('seed', 237),
                   'vid_res': params.get('vid_res', [720, 640]) #default video resolution [width, height]
                   }

    rewards = record_rollouts(task_id=params['task_id'],
                policy_params = policy_params,
                actor = actor,
                logdir=args.logdir,
                num_rollouts=args.num_rollouts,
                rollout_length = params['rollout_length'],
                shift = params['shift'])
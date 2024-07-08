import gymnasium as gym
import numpy as np

def instantiate_vec_gym_env(task_id, policy_params, num_envs):
    task_name = task_id.split('-')[0]

    if task_name == 'FrankaKitchen':
        pass
    elif 'Fetch' in task_name:
        env = gym.vector.make(task_id, num_envs = num_envs, max_episode_steps = policy_params['rollout_length'], reward_type = policy_params['reward_type'], render_mode = policy_params.get('render_mode', None), width = policy_params.get('vid_res', [0])[0], height = policy_params.get('vid_res', [0, 0])[1])
    elif 'HandManipulate' in task_name:
        env = gym.vector.make(task_id, num_envs = num_envs, max_episode_steps = policy_params['rollout_length'], reward_type = policy_params['reward_type'], render_mode = policy_params.get('render_mode', None), width = policy_params.get('vid_res', [0])[0], height = policy_params.get('vid_res', [0, 0])[1])
    else:
        env = gym.vector.make(task_id, num_envs = num_envs, render_mode = policy_params.get('render_mode', None), width = policy_params.get('vid_res', [0])[0], height = policy_params.get('vid_res', [0, 0])[1])
        
    env.reset(seed = policy_params['seed'])

    return env

def instantiate_gym_env(task_id, policy_params):
    task_name = task_id.split('-')[0]

    if task_name == 'FrankaKitchen':
        pass
    elif 'Fetch' in task_name:
        env = gym.make(task_id, max_episode_steps = policy_params['rollout_length'], reward_type = policy_params['reward_type'], render_mode = policy_params.get('render_mode', None), width = policy_params.get('vid_res', [0])[0], height = policy_params.get('vid_res', [0, 0])[1])
    elif 'HandManipulate' in task_name:
        env = gym.make(task_id, max_episode_steps = policy_params['rollout_length'], reward_type = policy_params['reward_type'], render_mode = policy_params.get('render_mode', None), width = policy_params.get('vid_res', [0])[0], height = policy_params.get('vid_res', [0, 0])[1])
    else:
        env = gym.make(task_id, render_mode = policy_params.get('render_mode', None), width = policy_params.get('vid_res', [0])[0], height = policy_params.get('vid_res', [0, 0])[1])
        
    env.reset(seed = policy_params['seed'])

    return env

def instantiate_gym_envs(policy_params):
    """
    Helper function to instantiate gym environments. Mainly handles different env hyperparameters from different env suites.
    """
    task_name = policy_params['task_id'].split('-')[0]

    if task_name == 'FrankaKitchen':
        pass
    elif 'Fetch' in task_name:
        env = gym.vector.make(policy_params['task_id'], num_envs = policy_params['num_envs'], max_episode_steps = policy_params['rollout_length'],  reward_type = policy_params['reward_type'])
        eval_env = gym.vector.make(policy_params['task_id'], num_envs = policy_params['num_eval_rollouts'], max_episode_steps = policy_params['rollout_length'],  reward_type = policy_params['reward_type'] )
    elif 'HandManipulate' in task_name:
        env = gym.vector.make(policy_params['task_id'], num_envs = policy_params['num_envs'], max_episode_steps = policy_params['rollout_length'],  reward_type = policy_params['reward_type'])
        eval_env = gym.vector.make(policy_params['task_id'], num_envs = policy_params['num_eval_rollouts'], max_episode_steps = policy_params['rollout_length'],  reward_type = policy_params['reward_type'] )
    else:
        env = gym.vector.make(policy_params['task_id'], num_envs = policy_params['num_envs'])
        eval_env = gym.vector.make(policy_params['task_id'], num_envs = policy_params['num_eval_rollouts'])
        
    env.reset(seed = policy_params['seed'])
    eval_env.reset(seed = policy_params['seed'])

    return env, eval_env

def handle_extra_params(params, extra_env_params):
    """
    A helper function used to extract relevant env hyperparams from all params parsed with ArgParse. 
    Modifies extra_env_params in-place.
    """

    task_name = params['task_id'].split('-')[0]
    print(task_name)
    if task_name == 'FrankaKitchen':
        pass
    elif 'Fetch' in task_name:
        extra_env_params['rollout_length'] = params.get('rollout_length', 50)
        extra_env_params['reward_type'] = params.get('reward_type', 'dense') #dense or sparse
    elif 'HandManipulate' in task_name:
        extra_env_params['rollout_length'] = params.get('rollout_length', 50)
        extra_env_params['reward_type'] = params.get('reward_type', 'dense') #dense or sparse

def get_state_pos_and_vel_idx(task_name):
    '''
		Helper function that returns all the state position and velocity indices. Used to instantiate the koopman policy actor network.
        
        Parameters
			task_name - the name of the relevant task.

        Returns
			state_pos_idx - an (ordered) list of position indices in order of corresponding variable in the action dimension
            state_vel_idx - an (ordered) list of velocity indices in order of corresponding variable in the action dimension
            
    '''
    task_name = task_name.split('-')[0] #remove the v[x] 
    state_pos_idx, state_vel_idx = None, None
    
	#Relevant docs for mujoco tasks - https://www.gymlibrary.dev/environments/mujoco/
    if task_name == 'Swimmer':
        state_pos_idx = np.r_[1:3]
        state_vel_idx = np.r_[6:8]
    elif task_name == 'Hopper':
        state_pos_idx = np.r_[2:5]
        state_vel_idx = np.r_[8:11]
    elif task_name == 'HalfCheetah':
        state_pos_idx = np.r_[2:8]
        state_vel_idx = np.r_[11:17]
    elif task_name == 'Walker2d':
        state_pos_idx = np.r_[2:8]
        state_vel_idx = np.r_[11:17]
    elif task_name == 'Ant':
        state_pos_idx = np.r_[5:13]
        state_vel_idx = np.r_[19:27]
    elif task_name == 'Humanoid':
        state_pos_idx = np.r_[6, 5, 7 : 22]
        state_vel_idx = np.r_[29, 28, 30 : 45]
    elif task_name == 'FrankaKitchen':
        # https://robotics.farama.org/envs/franka_kitchen/franka_kitchen/
        state_pos_idx = np.r_[0 : 9]
        state_vel_idx = np.r_[9 : 18]
    elif 'HandManipulate' in task_name:
        # https://robotics.farama.org/envs/shadow_dexterous_hand/manipulate_egg/ - this applies to the rest of the handmanipulate tasks
        state_pos_idx = np.r_[0 : 5, 6 : 9, 10 : 13, 14 : 18, 19 : 24]
        state_vel_idx = np.r_[24 : 29, 30 : 33, 34 : 37, 38 : 42, 43 : 48]
    elif task_name == 'FetchReach':
        # https://robotics.farama.org/envs/fetch/reach/
        state_pos_idx = np.r_[0 : 4]
        state_vel_idx = np.r_[5 : 9]        
    elif 'Fetch' in task_name:
        # https://robotics.farama.org/envs/fetch/
        state_pos_idx = np.r_[0 : 3, 9]
        state_vel_idx = np.r_[20 : 24]
    else:
        state_pos_idx = np.r_[:]
        state_vel_idx = np.r_[:]
    return state_pos_idx, state_vel_idx

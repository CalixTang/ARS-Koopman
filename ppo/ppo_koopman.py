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
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal

import os

#check if the dependency on logz (in ARS package) is what we want
from ARS import logz
from ARS.graph_results import graph_training_and_eval_rewards

from ppo_policies import TruncatedKoopmanNetworkPolicy, MinKoopmanNetworkPolicy, NNPolicy, get_policy
from torch_observables import LocomotionObservableTorch, LargeManipulationObservableTorch

class PPO:
    """
        This is the PPO class we will use as our model in main.py
    """
    def __init__(self, actor, critic, task_id, hyperparameters, extra_env_params = dict()):
        """
            Initializes the PPO model, including hyperparameters.

            Parameters:
                actor - the actor policy in an actor-critic setup
                critic - the critic policy in an actor-critic setup
                task_id - the id of the task_id for env
                hyperparameters - all extra arguments passed into PPO that should be hyperparameters.
                extra_env_params - extra parameters passed into environments. by default, none
            Returns:
                None
        """
        self.task_id = task_id
        #dummy env 
        dummy_env = gym.make(self.task_id)

        # Make sure the environment is compatible with our code
        # assert(type(dummy_env.observation_space) == gym.spaces.Box)
        assert(type(dummy_env.observation_space) == gym.spaces.Box or isinstance(dummy_env.observation_space, gym.spaces.dict.Dict))
        assert(type(dummy_env.action_space) == gym.spaces.Box)
        
        # Initialize hyperparameters for training with PPO
        self._init_hyperparameters(hyperparameters)

        logz.configure_output_dir(self.log_dir)

        # Extract environment information      
        if isinstance(dummy_env.observation_space, gym.spaces.dict.Dict):
            self.obs_dim = sum([v.shape[0] for k, v in dummy_env.observation_space.items()])
        else:   
            self.obs_dim = dummy_env.observation_space.shape[0]
        self.act_dim = dummy_env.action_space.shape[0]

        # Initialize actual envs
        self.env, self.eval_env = self.instantiate_gym_envs(extra_env_params)

        # Initialize actor and critic networks
        self.actor = actor                                                 # ALG STEP 1
        self.critic = critic

        # Initialize optimizers for actor and critic
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # Initialize the covariance matrix used to query the actor for actions
        
        if self.act_dim == len(self.actor_noise):
            self.cov_var = torch.tensor(self.actor_noise) 
        else:
            #hack - take the first val in the array for now... 
            self.cov_var = torch.full(size=(self.act_dim,), fill_value = self.actor_noise[0], dtype = torch.float32) #originally 0.5 
        self.cov_mat = torch.diag(self.cov_var)

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
            'start_time': 0,
            'total_time': 0,
            'delta_t': time.time_ns(),
            't_so_far': 0,          # timesteps so far
            'i_so_far': 0,          # iterations so far
            'batch_lens': [],       # episodic lengths in batch
            'batch_rews': [],       # episodic returns in batch
            'actor_losses': [],     # losses of actor network in current iteration
            'critic_losses': [],
            'lr': 0,
        }

    def learn(self, total_timesteps):
        """
            Train the actor and critic networks. Here is where the main PPO algorithm resides.

            Parameters:
                total_timesteps - the total number of timesteps to train for

            Return:
                None
        """
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        self.logger['start_time'] = time.time()
        t_so_far = 0 # Timesteps simulated so far
        i_so_far = 0 # Iterations ran so far

        best_mean_rew, best_koopman_mat = float('-inf'), None
        train_ep_rew = []
        eval_ep_rew = []

        while t_so_far < total_timesteps:                                                                       # ALG STEP 2
            # Autobots, roll out (just kidding, we're collecting our batch simulations here)
            batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones = self.rollout()                     # ALG STEP 3
            # Calculate advantage using GAE
            A_k = self.calculate_gae(batch_rews, batch_vals, batch_dones) 
            V = self.critic(batch_obs).squeeze()
            batch_rtgs = A_k + V.detach()   
            
            # Calculate how many timesteps we collected this batch
            t_so_far += batch_lens.sum()

            # Increment the number of iterations
            i_so_far += 1

            # Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
            # isn't theoretically necessary, but in practice it decreases the variance of 
            # our advantages and makes convergence much more stable and faster. I added this because
            # solving some environments was too unstable without it.
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # This is the loop where we update our network for some n epochs
            step = batch_obs.size(0)
            inds = np.arange(step)
            minibatch_size = step // self.num_minibatches
            loss = []
            critic_losses = []

            for _ in range(self.n_updates_per_iteration):                                                       # ALG STEP 6 & 7
                # Learning Rate Annealing
                frac = (t_so_far - 1.0) / total_timesteps
                new_lr = self.lr * (1.0 - frac)

                # Make sure learning rate doesn't go below 0
                new_lr = max(new_lr, 0.0)
                self.actor_optim.param_groups[0]["lr"] = new_lr
                self.critic_optim.param_groups[0]["lr"] = new_lr
                # Log learning rate
                self.logger['lr'] = new_lr

                # Mini-batch Update
                np.random.shuffle(inds) # Shuffling the index
                for start in range(0, step, minibatch_size):
                    end = start + minibatch_size
                    idx = inds[start:end]
                    # Extract data at the sampled indices
                    mini_obs = batch_obs[idx]
                    mini_acts = batch_acts[idx]
                    mini_log_prob = batch_log_probs[idx]
                    mini_advantage = A_k[idx]
                    mini_rtgs = batch_rtgs[idx]

                    # Calculate V_phi and pi_theta(a_t | s_t) and entropy
                    V, curr_log_probs, entropy = self.evaluate(mini_obs, mini_acts)

                    # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                    # NOTE: we just subtract the logs, which is the same as
                    # dividing the values and then canceling the log with e^log.
                    # For why we use log probabilities instead of actual probabilities,
                    # here's a great explanation: 
                    # https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
                    # TL;DR makes gradient descent easier behind the scenes.
                    logratios = curr_log_probs - mini_log_prob
                    ratios = torch.exp(logratios)
                    approx_kl = ((ratios - 1) - logratios).mean()

                    # Calculate surrogate losses.
                    surr1 = ratios * mini_advantage
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * mini_advantage

                    # Calculate actor and critic losses.
                    # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
                    # the performance function, but Adam minimizes the loss. So minimizing the negative
                    # performance function maximizes it.
                    actor_loss = (-torch.min(surr1, surr2)).mean()
                    critic_loss = nn.MSELoss()(V, mini_rtgs)

                    # Entropy Regularization
                    entropy_loss = entropy.mean()
                    # Discount entropy loss by given coefficient
                    actor_loss = actor_loss - self.ent_coef * entropy_loss                    
                    
                    # Calculate gradients and perform backward propagation for actor network
                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    # Gradient Clipping with given threshold
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.actor_optim.step()
                    # print(torch.norm(self.actor.koopman_layer.weight.grad))

                    # Calculate gradients and perform backward propagation for critic network
                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optim.step()

                    loss.append(actor_loss.detach())
                    critic_losses.append(critic_loss.detach())
                # Approximating KL Divergence
                if approx_kl > self.target_kl:
                    break # if kl aboves threshold
            # Log actor loss
            avg_loss = sum(loss) / len(loss)
            avg_critic_loss = sum(critic_losses) / len(critic_losses)
            self.logger['actor_losses'].append(avg_loss)
            self.logger['critic_losses'].append(avg_critic_loss)

            # Log a summary of our training so far
            self._log_summary()

            # add to training rew
            ep_rews = self.conv_batch_rews_to_ep_rews(batch_rews)
            train_ep_rew.append(ep_rews)
            if ep_rews.mean() > best_mean_rew:
                best_mean_rew = ep_rews.mean()
                best_koopman_mat = self.actor.get_koopman_matrix()

            # evaluation
            if i_so_far % self.eval_freq == 0:
                batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones = self.rollout(eval = True)

                #batch rews, batch_vals, and batch_dones are List[List[Tensor(num_envs)]] - list of (episodes as a list of (timestep -> tensor[num_envs]))
                
                #use helper func to translate batch rews to ep rews
                ep_rews = self.conv_batch_rews_to_ep_rews(batch_rews).detach()
                mean_rew, std_rew, min_rew, max_rew = ep_rews.mean(), ep_rews.std(), ep_rews.min(), ep_rews.max()

                #save ep rewards
                eval_ep_rew.append(ep_rews)

                logz.log_tabular("Time", time.time() - self.logger['start_time'])
                logz.log_tabular("Iteration", i_so_far)
                logz.log_tabular("AverageReward", mean_rew)
                logz.log_tabular("StdRewards", std_rew)
                logz.log_tabular("MaxRewardRollout", max_rew)
                logz.log_tabular("MinRewardRollout", min_rew)
                logz.log_tabular("timesteps", t_so_far)
                logz.dump_tabular()


                #save the actor and critic networks
                torch.save(self.actor.state_dict(), os.path.join(self.log_dir, 'ppo_actor.pth'))
                torch.save(self.critic.state_dict(), os.path.join(self.log_dir, 'ppo_critic.pth'))
                np.save(os.path.join(self.log_dir, "latest_koopman_policy_weights.npy"), self.actor.get_koopman_matrix())

        # save all ep rewards
        train_ep_rew = torch.cat(train_ep_rew, axis = 0).squeeze().detach().numpy()
        eval_ep_rew = torch.stack(eval_ep_rew, axis = 0).squeeze().detach().numpy()
        np.save(os.path.join(self.log_dir, "train_rewards.npy"), train_ep_rew)
        np.save(os.path.join(self.log_dir, "eval_rewards.npy"), eval_ep_rew)

        graph_training_and_eval_rewards(train_ep_rew, eval_ep_rew, self.log_dir, False)

        #save best koopman mat
        np.save(os.path.join(self.log_dir, "best_koopman_policy_weights.npy"), best_koopman_mat)


    def conv_batch_rews_to_ep_rews(self, batch_rews):
        '''
            Convert batch_rews List[List[tensor(num_envs)]] to tensor(episodes, num_envs) with cumulative reward per episode

            Parameters
                batch_rews - a List[List[tensor(num_envs)]] containing rewards at each episode at each timestep for each parallel environment.
            Returns
                ep_rews - a tensor(episodes, num_envs) containing cumulative reward per episode for each parallel environment.
        '''
        ep_rews = torch.zeros((len(batch_rews), batch_rews[0][0].shape[0]))
        for i in range(len(batch_rews)):
            #episode rewards (time dim, env dim)
            episode = batch_rews[i]
            episode = torch.stack(episode, axis = 0)
            ep_rews[i, :] = episode.sum(axis = 0)

        return ep_rews

    
    def calculate_gae(self, rewards, values, dones):
        batch_advantages = []  # List to store computed advantages for each timestep

        # Iterate over each episode's rewards, values, and done flags
        for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
            advantages = []  # List to store advantages for the current episode
            last_advantage = 0  # Initialize the last computed advantage

            # Calculate episode advantage in reverse order (from last timestep to first)
            for t in reversed(range(len(ep_rews))):
                if t + 1 < len(ep_rews):
                    # Calculate the temporal difference (TD) error for the current timestep
                    delta = ep_rews[t] + self.gamma * ep_vals[t+1] * (1 - ep_dones[t+1]) - ep_vals[t]
                else:
                    # Special case at the boundary (last timestep)
                    delta = ep_rews[t] - ep_vals[t]

                # Calculate Generalized Advantage Estimation (GAE) for the current timestep
                advantage = delta + self.gamma * self.lam * (1 - ep_dones[t]) * last_advantage
                last_advantage = advantage  # Update the last advantage for the next timestep
                advantages.insert(0, advantage)  # Insert advantage at the beginning of the list

            #turn advantages into the proper shape used in indexing during training
            advantages = torch.cat(advantages, axis = 0)

            # Extend the batch_advantages list with advantages computed for the current episode
            batch_advantages.extend(advantages)

        # This formulation seems wrong... Eps might have variable lens TODO verify
        # Convert the batch_advantages list to a PyTorch tensor of type float
        # return torch.tensor(batch_advantages, dtype=torch.float)

        # Convert batch_adv list to a (total T * num_envs, 1) tensor of type float
        batch_advantages = torch.tensor(batch_advantages, dtype = torch.float32)
        return batch_advantages


    def rollout(self, eval = False):
       
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_lens = []
        batch_vals = []
        batch_dones = []

        # Episodic data. Keeps track of rewards per episode, will get cleared
        # upon each new episode
        ep_rews = []
        ep_vals = []
        ep_dones = []
        t = 0 # Keeps track of how many timesteps we've run so far this batch

        #choose timestep cap for training vs eval (force only running 1 episode per env during eval)
        timestep_cap = self.timesteps_per_batch if not eval else 1
        num_envs = self.num_envs if not eval else self.num_eval_rollouts

        # Keep simulating until we've run more than or equal to specified timesteps per batch
        while t < timestep_cap:

            #choose the correct env to use for rollouts
            env = self.env if not eval else self.eval_env

            ep_rews = [] # rewards collected per episode
            ep_vals = [] # state values collected per episode
            ep_dones = [] # done flag collected per episode
            # Reset the environment. Note that obs is short for observation. 
            obs, _ = env.reset()

            if isinstance(obs, dict):
               obs = np.concatenate([obs['observation'], obs['achieved_goal'], obs['desired_goal']], axis = -1)

            obs = torch.from_numpy(obs).float()
            # Initially, envs are not done
            dones = torch.zeros((self.num_envs if not eval else self.num_eval_rollouts, )).float()

            # Run an episode for a maximum of max_timesteps_per_episode timesteps
            for ep_t in range(self.max_timesteps_per_episode):
                # If render is specified, render the environment
                if self.render:
                    env.render()

                t += num_envs # Increment timesteps ran this batch so far

                # Track observations in this batch
                batch_obs.append(obs)

                # Track done flags for all envs
                ep_dones.append(dones)

                # Calculate action and make a step in the env. 
                # Note that rew is short for reward.
                action, log_prob = self.get_action(obs, eval = eval)
                val = self.critic(obs) if not eval else np.zeros((1,)) #just to save some time in eval

                obs, rews, terminateds, dones, infos = env.step(action)

                if isinstance(obs, dict):
                    obs = np.concatenate([obs['observation'], obs['achieved_goal'], obs['desired_goal']], axis = -1)
                
                obs, rews, dones, terminateds = torch.from_numpy(obs).float(), torch.from_numpy(rews).float(), torch.from_numpy(dones).float(), torch.from_numpy(terminateds)

                #important - only shift the reward if this is not an eval rollout (shifting is only meant to help with training)
                if not eval:
                    rews -= self.shift

                # Track recent reward, action, and action log probability
                ep_rews.append(rews)
                ep_vals.append(val.flatten())
                batch_acts.append(torch.from_numpy(action).float())
                batch_log_probs.append(log_prob)

                # If the environment tells us ALL episodes are done, break
                if torch.all(dones) or torch.all(terminateds):
                    break

            # Track episodic lengths, rewards, state values, and done flags
            batch_lens.append((ep_t + 1) * torch.ones((num_envs, )))
            batch_rews.append(ep_rews)
            batch_vals.append(ep_vals)
            batch_dones.append(ep_dones)

        batch_lens = torch.stack(batch_lens, axis = 0)
        # Reshape data as tensors
        batch_obs = torch.cat(batch_obs, axis = 0) #(total timesteps, obs dim)
        batch_acts = torch.cat(batch_acts, axis = 0) #(total timesteps, act dim)
        batch_log_probs = torch.cat(batch_log_probs, axis = 0) # (total timesteps, 1?)

        # Log the episodic returns and episodic lengths in this batch.
        if not eval:
            self.logger['batch_rews'] = batch_rews
            self.logger['batch_lens'] = batch_lens

        # Here, we return the batch_rews instead of batch_rtgs for later calculation of GAE
        return batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones

    def get_action(self, obs, eval = False):
        """
            Queries an action from the actor network, should be called from rollout.

            Parameters:
                obs - the observation at the current timestep
                eval - whether or not this should be an eval timestep. If eval, don't sample noise.

            Return:
                action - the action to take, as a numpy array
                log_prob - the log probability of the selected action in the distribution
        """
        # Query the actor network for a mean action
        mean = self.actor(obs)

        # If we're testing, just return the deterministic action. Sampling should only be for training as our "exploration" factor.
        # Moved this before sampling to make code efficient
        if eval or self.deterministic:
            return mean.detach().numpy(), torch.tensor([1.])

        # Create a distribution with the mean action and std from the covariance matrix above.
        # For more information on how this distribution works, check out Andrew Ng's lecture on it:
        # https://www.youtube.com/watch?v=JjB58InuTqM
        dist = MultivariateNormal(mean, self.cov_mat)

        # Sample an action from the distribution
        action = dist.sample()

        # Calculate the log probability for that action
        log_prob = dist.log_prob(action)


        # Return the sampled action and the log probability of that action in our distribution
        return action.detach().numpy(), log_prob.detach()

    def evaluate(self, batch_obs, batch_acts):
        """
            Estimate the values of each observation, and the log probs of
            each action in the most recent batch with the most recent
            iteration of the actor network. Should be called from learn.

            Parameters:
                batch_obs - the observations from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of observation)
                batch_acts - the actions from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of action)
                batch_rtgs - the rewards-to-go calculated in the most recently collected
                                batch as a tensor. Shape: (number of timesteps in batch)
        """
        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        # if batch_obs.size(0) == 1:
        #     V = self.critic(batch_obs)
        # else:
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in get_action()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, log_probs, dist.entropy()

    def _init_hyperparameters(self, hyperparameters):
        """
            Initialize default and custom values for hyperparameters

            Parameters:
                hyperparameters - the extra arguments included when creating the PPO model, should only include
                                    hyperparameters defined below with custom values.

            Return:
                None
        """
        # Initialize default values for hyperparameters
        # Algorithm hyperparameters
        self.timesteps_per_batch = hyperparameters.get('timesteps_per_batch', 4800)                 # Number of timesteps to run per batch
        self.max_timesteps_per_episode = hyperparameters.get('max_timesteps_per_episode', 1600)           # Max number of timesteps per episode
        self.n_updates_per_iteration = hyperparameters.get('n_updates_per_iteration', 5)                # Number of times to update actor/critic per iteration
        self.lr = hyperparameters.get('lr', 0.005)                                 # Learning rate of actor optimizer
        self.gamma = hyperparameters.get('gamma', 0.95)                               # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = hyperparameters.get('clip', 0.2)                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA
        self.lam = hyperparameters.get('lam', 0.98)                                 # Lambda Parameter for GAE 
        self.num_minibatches = hyperparameters.get('num_minibatches', 6)                        # Number of mini-batches for Mini-batch Update
        self.ent_coef = hyperparameters.get('ent_coef', 0)                               # Entropy coefficient for Entropy Regularization
        self.target_kl = hyperparameters.get('target_kl', 0.02)                           # KL Divergence threshold
        self.max_grad_norm = hyperparameters.get('max_grad_norm', 0.5)                        # Gradient Clipping threshold
        self.actor_noise = hyperparameters.get('actor_noise', [0.2])                      #default noise to add to actor movement in training
        print(self.actor_noise)

        # Miscellaneous parameters
        self.render = hyperparameters.get('render', False)                             # If we should render during rollout
        self.eval_freq = hyperparameters.get('eval_freq', 10)                             # How often we save in number of iterations
        self.deterministic = hyperparameters.get('deterministic', False)                      # If we're testing, don't sample actions
        self.seed = hyperparameters.get('seed', None)								# Sets the seed of our program, used for reproducibility of results
        self.shift = hyperparameters.get('shift', 0)                                  # Shifts the reward in a direction during training to dissaude learning policies that do nothing (from ARS)

        self.log_dir = hyperparameters.get('log_dir', 'data')                           # The directory to save this run's logs, models, and figures to
        self.num_envs = hyperparameters.get('num_envs', 1)                               # The number of environments for training
        self.num_eval_rollouts = hyperparameters.get('num_eval_rollouts', 100)                    # The number of evaluation rollouts to use 

        
        # Sets the seed if specified
        if self.seed != None:
            # Check if our seed is valid first
            assert(type(self.seed) == int)

            # Set the seed 
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

    def instantiate_gym_envs(self, extra_params):
        """
        Helper function to instantiate gym environments. Mainly handles different env hyperparameters from different env suites.
        """
        task_name = self.task_id.split('-')[0]

        if task_name == 'FrankaKitchen':
            pass
        elif 'Fetch' in task_name:
            env = gym.vector.make(self.task_id, num_envs = self.num_envs, max_episode_steps = extra_params['max_timesteps_per_episode'],  reward_type = extra_params['reward_type'])
            eval_env = gym.vector.make(self.task_id, num_envs = self.num_eval_rollouts, max_episode_steps = extra_params['max_timesteps_per_episode'],  reward_type = extra_params['reward_type'] )
        elif 'HandManipulate' in task_name:
            env = gym.vector.make(self.task_id, num_envs = self.num_envs, max_episode_steps = extra_params['max_timesteps_per_episode'],  reward_type = extra_params['reward_type'])
            eval_env = gym.vector.make(self.task_id, num_envs = self.num_eval_rollouts, max_episode_steps = extra_params['max_timesteps_per_episode'],  reward_type = extra_params['reward_type'] )
        else:
            env = gym.vector.make(self.task_id, num_envs = self.num_envs)
            eval_env = gym.vector.make(self.task_id, num_envs = self.num_eval_rollouts)
            
        env.reset(seed = self.seed)
        eval_env.reset(seed = self.seed)

        return env, eval_env

    def _log_summary(self):
        """
            Print to stdout what we've logged so far in the most recent batch.

            Parameters:
                None

            Return:
                None
        """
        # Calculate logging values. I use a few python shortcuts to calculate each value
        # without explaining since it's not too important to PPO; feel free to look it over,
        # and if you have any questions you can email me (look at bottom of README)

        #TODO: verify that the math in here actually calculates what we think it does
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        curr_t = time.time()
        elapsed_time = curr_t - self.logger['start_time']
        self.logger['total_time'] = elapsed_time

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        lr = self.logger['lr']
        avg_ep_lens = (self.logger['batch_lens'].mean())
        ep_rews = self.conv_batch_rews_to_ep_rews(self.logger['batch_rews']).detach()
        avg_ep_rews, std_ep_rews, max_ep_rews, min_ep_rews = ep_rews.mean(), ep_rews.std(), ep_rews.max(), ep_rews.min()
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])
        avg_critic_loss = np.mean([losses.float().mean() for losses in self.logger['critic_losses']])

        # Round decimal places for more aesthetic logging messages
        # avg_ep_lens = str(round(avg_ep_lens, 5))
        # avg_ep_rews = str(round(avg_ep_rews, 5))
        # avg_actor_loss = str(round(avg_actor_loss, 5))

        # Log data
        print("----------------")
        print("Iteration ", i_so_far)
        print("Number of Episodes", torch.numel(ep_rews))
        print("Elapsed Time ", elapsed_time)
        print("Average Episodic Length ", avg_ep_lens)
        print("Average Episodic Rewards ", avg_ep_rews)
        print("Standard Deviation of Episodic Rewards ", std_ep_rews)
        print("Max Episodic Reward ", max_ep_rews)
        print("Min Episodic Reward ", min_ep_rews)
        print("Average Actor Loss ", avg_actor_loss)
        print("Average Critic Loss ", avg_critic_loss)
        print("Current Elapsed Timesteps ", t_so_far)
        print("Iteration Time (secs) ", delta_t)
        print("Learning Rate ", lr)
        print("----------------", flush = True)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []


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
        extra_env_params['max_timesteps_per_episode'] = params.get('max_timesteps_per_episode', 50)
        extra_env_params['reward_type'] = params.get('reward_type', 'dense') #dense or sparse
    elif 'HandManipulate' in task_name:
        extra_env_params['max_timesteps_per_episode'] = params.get('max_timesteps_per_episode', 50)
        extra_env_params['reward_type'] = params.get('reward_type', 'dense') #dense or sparse


def run_ppo(params):
    print("Parameters: ", params)
    
    #set up logging directory
    dir_path = params['dir_path']
    logdir = None

    if not(os.path.exists(dir_path)):
        os.makedirs(dir_path)
    if params.get('run_name', None) is not None:
        logdir = os.path.join(dir_path, params['run_name'])
    else:
        logdir = os.path.join(dir_path, str(time.time_ns()))
        while os.path.exists(logdir):
            logdir = os.path.join(dir_path, str(time.time_ns()))
    print(f"Logging to directory {logdir}")
    os.makedirs(logdir)

    logz.configure_output_dir(logdir)
    logz.save_params(params)

    #set up ppo hyperparameters
    ppo_hyperparameters = {
        'timesteps_per_batch': params['timesteps_per_batch'],
        'max_timesteps_per_episode': params['max_timesteps_per_episode'],
        'n_updates_per_iteration': params['n_updates_per_iteration'],
        'lr': params['ppo_lr'],
        'gamma': params['ppo_gamma'],
        'clip': params['clip'], 
        'lam': params['lam'], 
        'num_minibatches': params['num_minibatches'],
        'ent_coef': params['ent_coef'],
        'target_kl': params['target_kl'],
        'max_grad_norm': params['max_grad_norm'], 
        'shift': params['shift'],
        'seed': params['seed'],
        'log_dir': logdir,
        'task_id': params['task_id'],
        'num_envs': params['num_envs'],
        'num_eval_rollouts': params['num_eval_rollouts'],
        'actor_noise': params['actor_noise']
    }
    
    #set up dummy environment to extract shape info
    env = gym.make(params['task_id'])

    #set up actor network (Koopman Policy)
    obs_dim = 0
    if isinstance(env.observation_space, gym.spaces.dict.Dict):
        obs_dim = sum([v.shape[0] for k, v in env.observation_space.items()])
    else:
        obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    state_pos_idx, state_vel_idx = get_state_pos_and_vel_idx(params['task_id'])

    #set up actor policy
    actor = get_policy(params['policy_type'], obs_dim, act_dim, params['lifting_function'], state_pos_idx, state_vel_idx, params['PDctrl_P'], params['PDctrl_D'])
    
	#set up critic network (obs -> val)
    critic = NNPolicy(obs_dim, 1)

    extra_env_params = {}
    handle_extra_params(params, extra_env_params)

	#instantitate ppo object with params
    ppo = PPO(actor, critic, params['task_id'], ppo_hyperparameters, extra_env_params = extra_env_params)
    
	#run ppo learn
    ppo.learn(params['total_timesteps'])
    
	#TODO: save koopman policy weights, actor and critic networks as pytorch models, and any relevant figures

    return

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    #basic arguments
    parser.add_argument('--task_id', type=str, default='Swimmer-v4') #any mujoco env v4
    

	#PPO arguments
    parser.add_argument('--total_timesteps', type = int, default = 1e8) #number of total timesteps to allot to ppo
    parser.add_argument('--num_envs', type = int, default = 5) #number of parallel environments
    #I'm going to use reward shift like ARS does because it seems useful for training
    parser.add_argument('--timesteps_per_batch', type = int, default = 5000) # Number of timesteps to run per batch
    parser.add_argument('--max_timesteps_per_episode', type = int, default = 1000) # Max number of timesteps per episode (max rollout length) - important: different env suites will have different normal values
    parser.add_argument('--n_updates_per_iteration', type = int, default = 5) # Number of times to update actor/critic per iteration
    parser.add_argument('--ppo_lr', type = float, default = 5e-3) #LR for actor
    parser.add_argument('--ppo_gamma', type = float, default = 0.95) #reward discount factor
    parser.add_argument('--clip', type = float, default = 0.2) #ppo clipping epsilon (1 + eps, 1 - eps).
    #PPO2 args
    parser.add_argument('--lam', type = float, default = 0.98) #lambda param for GAE 
    parser.add_argument('--num_minibatches', type = int, default = 6) # Number of mini-batches for Mini-batch Update 
    parser.add_argument('--ent_coef', type = float, default = 0) # Entropy coefficient for Entropy Regularization 
    parser.add_argument('--target_kl', type = float, default = 0.02) # KL Divergence threshold 
    parser.add_argument('--max_grad_norm', type = float, default = 0.5) # Gradient Clipping threshold 
    parser.add_argument('--actor_noise', type = float, default = [0.2], nargs='*') #the noise to allow for the actor actions
    parser.add_argument('--lifting_function', default = "Locomotion") #the noise to allow for the actor actions
    # for Swimmer-v1 and HalfCheetah-v1 use shift = 0
    # for Hopper-v1, Walker2d-v1, and Ant-v1 use shift = 1
    # for Humanoid-v1 used shift = 5
    parser.add_argument('--shift', type=float, default = 0) #TODO: tweak as necessary
    parser.add_argument('--seed', type = int) #random seed	
    parser.add_argument('--num_eval_rollouts', type = int, default = 100) #the number of environment rollouts to perform during evaluation. probably leave it at 100 for consistency

    #PD controller params - adding for flexibility but probably won't change at all
    parser.add_argument('--PDctrl_P', type = float, default = 0.1)
    parser.add_argument('--PDctrl_D', type = float, default = 0.001)
    
    #utility arguments
    parser.add_argument('--policy_type', type = str, default = 'minkoopman')
    parser.add_argument('--dir_path', type=str, default='data') #the folder to save runs to
    parser.add_argument('--params_path', type = str, default = None)
    parser.add_argument('--policy_checkpoint_path', type = str)
    parser.add_argument('--run_name', type = str)

    #Robosuite env args
    parser.add_argument('--reward_type', type = str, default = 'dense', choices=['sparse', 'dense']) #the type of reward for robosuite environments. doesn't work with all envs, mostly just the dexterous manipulation ones.

    args = parser.parse_args()
    params = vars(args)

    if args.params_path is not None:
        import json
        params = json.load(open(args.params_path, 'r'))
        # print(params)
        if args.run_name:
            params['run_name'] = args.run_name

    run_ppo(params)
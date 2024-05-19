import pickle
import os
import numpy as np

with open('../Samples/Door/Door_task.pickle', 'rb') as infile:
    data = pickle.load(infile)
    #print(f"data is {data}")
    data_dict = data[0]
    print(list(data_dict.keys()))

    
    obs, obs_vis = data_dict['observations'], data_dict['observations_visualization']
    state_dict = dict()
    state_dict['handpos'] = np.array(obs_vis[:28])
    #state_dict['handvel'] = obs_visual[30:58]
    state_dict['objpos'] = np.array(obs[32:35])
    #state_dict['objvel'] = obs_visual[58:59]
    print(state_dict['handpos'].shape, state_dict['objpos'].shape)


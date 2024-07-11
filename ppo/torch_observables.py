import torch 

def get_observable(observable_name):
    observable_name = observable_name.lower()
    if observable_name == 'locomotion':
        return LocomotionObservableTorch
    elif observable_name == 'largemanipulation':
        return LargeManipulationObservableTorch
    raise NotImplementedError

class LocomotionObservableTorch():
    def __init__(self, obs_dim):
        self.obs_dim = obs_dim
    
    def z(self, obs):  
        """
        Lifts the environment state from state space to full "observable space' (Koopman). g(x) = z.
        Inputs: environment states
        Outputs: state in lifted space
        """

        #for consistency, I keep the same order of functions as DraftedObservable as used in KODex/CIMER

        obs = obs.detach().clone()
        #concatenate along the last dimension of obs - usually obs is either single vector (obs_dim,) or batched (batch_dim, obs_dim)
        return torch.cat((obs, obs ** 2, torch.sin(obs), torch.cos(obs)), axis = len(obs.shape) - 1)
    
    def compute_observables_from_self(self):
        """
        Observation functions: original states, original states^2, cross product of hand states
        """
        return 4 * self.obs_dim
    
"""
A larger observable because we might need it
"""
class LargeManipulationObservableTorch():
    def __init__(self, obs_dim):
        self.obs_dim = obs_dim

    def z(self, obs):
        """
        Lifts the environment state from state space to full "observable space' (Koopman). g(x) = z.
        Inputs: environment states
        Outputs: state in lifted space
        """

        obs = obs.detach().clone()
        if len(obs.shape) == 1:
            return torch.cat((obs, 
                              obs ** 2, 
                              torch.sin(obs), 
                              torch.cos(obs), 
                              (obs.unsqueeze(1) @ obs.unsqueeze(0))[torch.triu(torch.ones((obs.shape[0], obs.shape[0])), diagonal = 1).bool()], 
                              torch.flatten((obs ** 2).unsqueeze(0) * obs.unsqueeze(1))), 
                              axis = len(obs.shape) - 1)
        else:
            return torch.cat((obs, 
                              obs ** 2, 
                              torch.sin(obs), 
                              torch.cos(obs), 
                              torch.bmm(obs.unsqueeze(2), obs.unsqueeze(1))[:, torch.triu(torch.ones((obs.shape[1], obs.shape[1])), diagonal = 1).bool()], 
                              torch.flatten((obs ** 2).unsqueeze(1) * obs.unsqueeze(2), start_dim = 1)), 
                              axis = len(obs.shape) - 1)
    
    def compute_observables_from_self(self):
        return 4 * self.obs_dim + (self.obs_dim * (self.obs_dim - 1)) // 2 + self.obs_dim ** 2
import torch 

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
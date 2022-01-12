from torch import nn
from torch import optim
import torch

import numpy as np

from Agents.Networks import DuelingNetwork

class DQN:
    def __init__(self, frame_hist, act_dim, n_steps, gamma=0.99, n_hid=64, lr=1e-4, epsilon=0.9, tau=1, device=None, clip_grad_val=None, network=None):
        '''
        Baseline implimentation of Q-Function:
        
        - Collect sample transitions from the environment (store in a replay memory)
        - sample batches of experience (from replay memory)
        - For each transition, calculate the 'target' value (current estimate of state-action value) 
                y_t = r_t + gamma * max_a Q(s_{t+1}, a)
        - Calculate the estimate of the state-action values.
                y_hat_t = Q(s_t, a_t)
        - Calculate the loss, L(y_hat_t, y_t) - m.s.e (td-error)
        - Compute the gradient of L with respect to the network parameters, and take a gradient descent step.        

        (Does not use target network for action evaluation.)
        '''
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

        self.act_dim = act_dim
        self.frame_hist = frame_hist
        self.gamma = gamma
        
        self.epsilon = epsilon
        self.tau = tau        
        self.epsilon_decay_rate = 0.9995
        self.tau_decay_rate = 0.9995
        self.min_epsilon = 0.05
        self.min_tau = 0.1
        
        self.n_hid = n_hid 
        self.lr = lr
        self.mse_loss = nn.MSELoss(reduction='none')
        
        self.clip_grad_val = clip_grad_val     
        self.n_steps = n_steps if n_steps is not None else 1
        self.init_network(network)
        
    def init_network(self, network=None):  
        if network is None:
            self.model = DuelingNetwork(self.frame_hist, self.act_dim, self.n_hid, clip_grad_val=self.clip_grad_val).to(self.device)
        else:
            self.model = network(self.frame_hist, self.act_dim, self.n_hid, clip_grad_val=self.clip_grad_val).to(self.device)

        self.optim = optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optim, gamma=1)
    
    def get_loss(self, experience, IS_weights):    
        states, actions = experience['states'], experience['actions']
        q_preds = self.model(states).gather(dim=-1, index=actions.long()).squeeze(-1)
        q_targets = self.compute_targets(experience)    
    
        q_loss = self.mse_loss(q_preds, q_targets)
        q_loss = torch.multiply(IS_weights, q_loss).mean()
        errors = (q_preds.detach() - q_targets).abs().cpu().numpy()
            
        return q_loss, errors
            
    def compute_targets(self, experience):        
        with torch.no_grad():
            q_preds_next = self.model(experience['next_states'])
            
        max_q_preds, _ = q_preds_next.max(dim=-1, keepdim=False)
        q_targets = experience['rewards'] + (self.gamma ** self.n_steps)*max_q_preds*(1 - experience['dones'])    
        return q_targets

    def train_network(self, experience, IS_weights):        
        # Compute the Q-targets and TD-Error
        
        avg_loss = 0.0
        loss, error = self.get_loss(experience, IS_weights)
        self.model.train_step(loss, self.optim, self.lr_scheduler)
        
        return loss.item(), error
    
    def act_epsilon_greedy(self, state):
        # With probability epsilon, take a random action 
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.act_dim)
        
        action_values = self.model(state).detach()
        action = action_values.argmax()
        return action.cpu().squeeze().numpy() 

    def act_boltzmann(self, state):      
        # sample from a Boltzman distribution over the state-action values
        logits = self.model(state) / self.tau
        action_pd = torch.distributions.Categorical(logits=logits)             
        action = action_pd.sample()
        return action.squeeze().numpy() 
    
    def act_greedy(self, state): 
        # Act greddily with respect to the action values
        action_values = self.model(state).detach()
        action = action_values.argmax()
        return action.cpu().squeeze().numpy()

    def act(self, state, policy='boltzmann'):
        if not torch.is_tensor(state):              
            state = torch.Tensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if policy=='boltzmann':
                return self.act_boltzmann(state)
            elif policy=='epsilon_greedy':
                return self.act_epsilon_greedy(state)
            else:
                return self.act_greedy(state) 
        
    
    def update(self):
        # update hyperparameters 
        self.epsilon = max(self.epsilon*self.epsilon_decay_rate, self.min_epsilon)
        self.tau = max(self.tau*self.tau_decay_rate, self.min_tau)

    def save(self, path):
        torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optim.state_dict()
                }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        
        
class DDQN(DQN):
    '''
        Double Deep Q-Networks: 
        Based on the extenstion to DQN from the paper:
            Deep Reinforcement Learning with Double Q-learning, 2015
            Hado van Hasselt and Arthur Guez and David Silver
            https://arxiv.org/pdf/1509.06461.pdf
            
        When calculating the training target for our Q-Network, we utilize a second network (the target network) to
        evaluate the actions selected by the 'online' network which is used to select actions.
        
        The new 'target' value calculation:  
                y_t = r_t + gamma * Q_target(s_{t+1}, argmax_a Q_online(s_{t+1}, a))
    '''
    def __init__(self, frame_hist, action_dim, gamma=0.99, epsilon=0.01, n_hid=64, lr=1e-4, device=None, noisy_networks=True, target_update_freq=100, clip_grad_val=None):
        self.target_update_freq = target_update_freq
        self.noisy_network = noisy_networks
        self.ratio = 0.25
        
        super().__init__(frame_hist, action_dim, gamma, epsilon=0.01, device=device, n_hid=n_hid, lr=lr, clip_grad_val=clip_grad_val)      
        
    def init_network(self):
        '''
        Initialize the neural network related objects used to learn the policy function
        '''
             
        self.model = DuelingNetwork(self.frame_hist, self.act_dim, self.n_hid, noisy=self.noisy_network, clip_grad_val=self.clip_grad_val).to(self.device)
        self.target = DuelingNetwork(self.frame_hist, self.act_dim, self.n_hid, noisy=self.noisy_network, clip_grad_val=self.clip_grad_val).to(self.device)
        self.target.polyak_update(source_network=self.model, source_ratio=1.0) # copy parameters from model to target
        
        # We do not need to compute the gradients of the target network. It will be periodically 
        # updated using the parameters in the online network.
        for param in self.target.parameters():
            param.requires_grad = False
        
        self.model.train()
        self.target.train()
        
        self.online_network = self.model          
        self.eval_network = self.target     
        
        self.optim = optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optim, gamma=1)
     
    def compute_targets(self, experience):        
        with torch.no_grad():
            online_q_preds = self.online_network(experience["next_states"]) # online network selects actions
            
            # Resample parameter noise if using noisy networks.
            if self.noisy_network: 
                self.eval_network.reset_noise()
                
            eval_q_preds = self.eval_network(experience["next_states"])     # Target network evaluates actions
        
        online_actions = online_q_preds.argmax(dim=-1, keepdim=True)
        next_q_preds = eval_q_preds.gather(-1, online_actions).squeeze(-1)        
        q_targets = experience['rewards'] + (self.gamma ** self.n_steps)*(1 - experience['dones'])*next_q_preds
        return q_targets

    def update_target_network(self, train_step):
        if train_step % self.target_update_freq == 0:
            self.target.polyak_update(source_network=self.model, source_ratio=self.ratio)            
    
    def reset_noise(self):
        self.online_network.reset_noise()
    
    def update(self):
        super().update()


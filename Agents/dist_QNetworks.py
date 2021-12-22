from torch import nn
from torch import optim
import torch

import numpy as np
from Agents.Networks import DuelingNetwork

class Cat_DDQN:
    def __init__(self, fram_hist, act_dim, atoms, V_min, V_max, n_steps, gamma=0.99, n_hid=64, lr=1e-4, epsilon=0.01, device=None, clip_grad_val=None, network=None):
        '''
            Implimentation of the Categorical DQN introduced in the paper:
                A Distributional Perspective on Reinforcement Learning, 2017
                https://arxiv.org/pdf/1707.06887.pdf
        '''
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

        # Categorical Distribution hyper params
        self.atoms = atoms
        self.V_min = V_min
        self.V_max = V_max
        self.support = torch.linspace(V_min, V_max, atoms).to(self.device)
        self.dz = (V_max - V_min) / (atoms - 1)

        # NN hyper parameters 
        self.act_dim = act_dim
        self.frame_hist = frame_hist
        self.gamma = gamma
        self.n_hid = n_hid 
        self.lr = lr

        # action_selection hyper parameters
        self.epsilon = epsilon
        self.epsilon_decay_rate = 0.9999
        self.min_epsilon = 0.001
        
        self.clip_grad_val = clip_grad_val     
        self.n_steps = n_steps
        self.init_network(network)
        
    def init_network(self, network=None):  
        if network is None:
            self.model = DuelingNetwork(self.frame_hist, self.act_dim, self.n_hid, clip_grad_val=self.clip_grad_val).to(self.device)
            self.target = DuelingNetwork(self.frame_hist, self.act_dim, self.n_hid, clip_grad_val=self.clip_grad_val).to(self.device)
        else:
            self.model = network(self.frame_hist, self.act_dim, self.n_hid, clip_grad_val=self.clip_grad_val).to(self.device)
            self.target = network(self.frame_hist, self.act_dim, self.n_hid, clip_grad_val=self.clip_grad_val).to(self.device)

        self.target.polyak_update(source_network=self.model, source_ratio=1.0) # copy parameters from model to target
        self.model.train()
        self.target.train()
        
        self.online_network = self.model          
        self.eval_network = self.target     

        self.optim = optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optim, gamma=1)
    
    def train_network(self, experience, IS_weights):        
        # Compute the Cross Entropy loss using the distribution of rewards and subsequent states

        loss, error = self.get_loss(experience, IS_weights)
        loss = (IS_weights * loss).mean()
        self.model.train_step(loss, self.optim, self.lr_scheduler)
        
        return loss.item(), error

    def get_loss(self, experience, IS_weights):    
        # Log probabilities of action-value dist. for Cross Entropy Loss 
        log_ps_x = torch.functional.log_softmax(self.online_network(experience['states']), dim=2) # log(x_t, ·; θ_online)
        log_ps_x_a = log_ps_x[:, experience['actions']] # log(x_t, a_t; θ_online)
        
        # compute the target distribution using next states and returns
        m = self.compute_targets(experience)
        # minimize distance between Z_θ and TZ_θ -> Cros entropy Loss (See Algorithm 1 from paper referenced above)
        loss = -torch.sum(m * log_ps_x_a, 1)

        return loss, loss.detach().abs().cpu().numpy()
            
    def compute_targets(self, experience):        
        batch_size = experience['states'].shape[0]

        with torch.no_grad():
            m = torch.zeros(batch_size*self.atoms).to(self.device).float()
            off_set = (
                    torch.linspace(0, (batch_size-1)*self.atoms, batch_size)
                    .unsqueeze(0)
                    .expand(batch_size, self.atoms)
                    .to(experience)
            ) # line up the value of Tz with the distribution of atoms.


            


            # Distribute the probabilities to each atom of Tz 
            m.index_add(0, (l + off_set).view(-1), (ps_x_a * (u - b)).view(-1)) # m_l += p(s_{t+n}, a*) * (u - b)
            m.index_add(0, (u + off_set).view(-1), (ps_x_a * (b - l)).view(-1)) # m_lu+= p(s_{t+n}, a*) * (b -l)
            m = torch.reshape(m, [batch_size, self.atome])

        return m

   
    def act_epsilon_greedy(self, state):
        # With probability epsilon, take a random action 
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.act_dim)
        
        return self.act_greedy(state)

    
    def act_greedy(self, state): 
        # advantage distribution for each action
        action_value_prob = torch.functional.softmax(self.model(state).detach(), dim=2) 
        # Compute Expected action values based on the value distributions
        expected_action_values = (action_prob * self.support).sum(2) 

        return expected_action_values.argmax(1).item()

    def act(self, state, policy='epsilon_greedy'):
        if not torch.is_tensor(state):              
            state = torch.Tensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if policy=='epsilon_greedy':
                return self.act_epsilon_greedy(state)
            else:
                return self.act_greedy(state) 
        
    
    def update(self, count):
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

    def reset_noise(self):
        # 
        self.model.reset_noise()
 

import os 
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from networks import ActorNetwork, CriticNetwork
from buffer import ReplayBuffer
from noise import OUActionNoise

class Agent():
    def __init__(self, alpha, beta, state_dims, tau, action_dims, gamma=0.99,
                 max_size=1000000, fc1_dims=400, fc2_dims=300,batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.batch_size = batch_size

        self.memory = ReplayBuffer(max_size, state_dims, action_dims)

        self.noise = OUActionNoise(mu=np.zeros(action_dims))

        self.behavior_actor = ActorNetwork(alpha, state_dims, fc1_dims, 
                                           fc2_dims, action_dims, 
                                           name='behavior_actor')
        
        self.behavior_critic = CriticNetwork(beta, state_dims, fc1_dims, 
                                             fc2_dims, action_dims, 
                                             name='behavior_critic')
        
        self.target_actor = ActorNetwork(alpha, state_dims, fc1_dims, 
                                         fc2_dims, action_dims, 
                                         name='target_actor')
        
        self.target_critic = CriticNetwork(beta, state_dims, fc1_dims,
                                           fc2_dims, action_dims,
                                           name='target_critic')
        
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        self.behavior_actor.eval()
        state = T.tensor([observation], dtype=T.float)\
            .to(self.behavior_actor.device)
        mu = self.behavior_actor.forward(state).to(self.behavior_actor.device)
        mu_prime = mu + T.tensor(self.noise(),
                                 dtype=T.float).to(self.behavior_actor.device)
        self.behavior_actor.train()

        return mu_prime.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = \
            self.memory.sample_buffer(self.batch_size)
        
        states = T.tensor(states, dtype=T.float)\
            .to(self.behavior_actor.device)
        actions = T.tensor(actions, dtype=T.float)\
            .to(self.behavior_actor.device)
        rewards = T.tensor(rewards, dtype=T.float)\
            .to(self.behavior_actor.device)
        next_states = T.tensor(next_states, dtype=T.float) \
            .to(self.behavior_actor.device)
        dones = T.tensor(dones, dtype=T.bool).to(self.behavior_actor.device)
        
        target_action = self.target_actor.forward(next_states)
        target_next_state_value = \
            self.target_critic.forward(next_states, target_action)
        behavior_state_value = self.behavior_critic.forward(states, actions)

        target_next_state_value[dones] = 0.0
        target_next_state_value = target_next_state_value.view(-1)
        
        target = rewards + self.gamma * target_next_state_value
        target = target.view(self.batch_size, 1)

        self.behavior_critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, behavior_state_value)
        critic_loss.backward()
        self.behavior_critic.optimizer.step()

        self.behavior_actor.optimizer.zero_grad()
        actor_loss = -self.behavior_critic.forward(states, 
                                        self.behavior_actor.forward(states))
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.behavior_actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        behavior_actor_params = self.behavior_actor.named_parameters()
        behavior_critic_params = self.behavior_critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        behavior_actor_state_dict = dict(behavior_actor_params)
        behavior_critic_state_dict = dict(behavior_critic_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_state_dict = dict(target_critic_params)

        for name in target_actor_state_dict:
            target_actor_state_dict[name] = tau * behavior_actor_state_dict[name].clone() \
                            + (1 - tau) * target_actor_state_dict[name].clone()
        for name in target_critic_state_dict:
            target_critic_state_dict[name] = tau * behavior_critic_state_dict[name].clone() \
                            + (1 - tau) * target_critic_state_dict[name].clone()
            
        self.target_actor.load_state_dict(target_actor_state_dict)
        self.target_critic.load_state_dict(target_critic_state_dict)

    def save_models(self):
        self.behavior_actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.behavior_critic.save_checkpoint()
        self.target_critic.save_checkpoint()
    
    def load_model(self):
        self.behavior_actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.behavior_critic.load_checkpoint()
        self.target_critic.load_checkpoint()




import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class NN_Q(nn.Module):
    def __init__(self, input_size, output_size):
        super(NN_Q, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.ac1 = nn.SiLU()
        self.fc2 = nn.Linear(64, 64)
        self.ac2 = nn.SiLU()
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, state):
        x = self.fc1(state)
        x = self.ac1(x)
        x = self.fc2(x)
        x = self.ac2(x)
        x = self.fc3(x)
        return x

class Agent_DQN:
    def __init__(self, env, discount=0.9, rep=20, batch_size=100, epsilon=0.1):
        '''Initializes the DQN agent'''
        self.env = env
        self.discount = discount
        self.rep = rep
        self.batch_size = batch_size
        self.epsilon = epsilon

        self.QNet = NN_Q(env.observation_space.shape[0] * env.observation_space.shape[1] + 1, 1)
        self.QNet_optimizer = torch.optim.Adam(self.QNet.parameters(), lr=5e-4)

        self.QNet_hat = NN_Q(env.observation_space.shape[0] * env.observation_space.shape[1] + 1, 1)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.QNet.to(self.device)
        self.QNet_hat.to(self.device)

    def act(self, state, exploration=False):
        '''Returns the action to take based on the given state'''
        q_act = {}
        state = state.to(self.device)
        with torch.no_grad():
            for i in self.env.unwrapped.get_available_actions():
                st = torch.cat((state, torch.tensor([i]).to(self.device))).to(self.device)
                q_act[i] = self.QNet(st).item()
            if np.random.rand() < self.epsilon and exploration:
                return np.random.choice(list(q_act.keys()))
            return max(q_act, key=q_act.get)

    def save_models(self, episode, path=''):
        '''Saves the models to the given path'''
        torch.save(self.QNet.state_dict(), f'{path}QNet_model_{episode}.pt')
        torch.save(self.QNet_hat.state_dict(), f'{path}QNetHat_model_{episode}.pt')

    def load_models(self, first_path, second_path):
        '''Loads the models from the given paths'''
        self.QNet.load_state_dict(torch.load(first_path, weights_only=True, map_location=torch.device(self.device)))
        self.QNet_hat.load_state_dict(torch.load(second_path, weights_only=True, map_location=torch.device(self.device)))
        self.QNet.eval()
        self.QNet_hat.eval()

    def learn(self, states, actions, rewards, next_states, dones):
        '''Updates the QNet based on the given states, actions, rewards, next_states, and dones'''
        states = states.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        actions = actions.to(self.device)
        dones = dones.to(self.device)
        for i in range(self.rep):
            indexes = torch.randperm(states.shape[0])[:min(self.batch_size, states.shape[0])].to(self.device)
            states = states[indexes]
            actions = actions[indexes]
            rewards = rewards[indexes]
            next_states = next_states[indexes]
            dones = dones[indexes]
            target = torch.empty([next_states.shape[0], 1]).to(self.device)
            with torch.no_grad():
                for j in range(next_states.shape[0]):
                    if dones[j]:
                      target[j] = rewards[j].float()
                    else:
                      next_actions = self.act(next_states[j])
                      target[j] = rewards[j].float() + self.discount * self.QNet_hat(torch.cat((next_states[j], torch.tensor([next_actions]).to(self.device))))
            input = torch.hstack([states, actions]).to(self.device)
            self.QNet_optimizer.zero_grad()
            output = self.QNet(input)
            loss = F.mse_loss(output, target).to(torch.float32)
            loss.backward()
            self.QNet_optimizer.step()
    
    def decrease_epsilon(self):
        '''Decreases the epsilon value'''
        self.epsilon = max(self.epsilon * 0.9, 0.1)
        return self.epsilon
    
    def update_target(self):
        '''Updates the target network'''
        self.QNet_hat.load_state_dict(self.QNet.state_dict())
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class NN_DuelQ(nn.Module):
    def __init__(self, input_size, output_size):
        '''Creates the model'''
        super(NN_DuelQ, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.ac1 = nn.SiLU()
        self.fc2 = nn.Linear(128, 128)
        self.ac2 = nn.SiLU()
        self.V = nn.Linear(128, 1)
        self.A = nn.Linear(128, output_size)

    def forward(self, state):
        '''Do the prediction given the input'''
        if state.dim() == 1:
            state = state.unsqueeze(0)
        x = self.fc1(state)
        x = self.ac1(x)
        x = self.fc2(x)
        x = self.ac2(x)
        V = self.V(x)
        A = self.A(x)
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q

class Agent_DuelDQN:
    def __init__(self, env, discount=0.9, rep=15, batch_size=128, epsilon=0.5):
        '''Initializes the agent'''
        self.env = env
        self.discount = discount
        self.rep = rep
        self.batch_size = batch_size
        self.epsilon = epsilon

        self.QNet = NN_DuelQ(env.observation_space.shape[0] * env.observation_space.shape[1], env.action_space.n)
        self.QNet_optimizer = torch.optim.Adam(self.QNet.parameters(), lr=5e-4)

        self.QNet_hat = NN_DuelQ(env.observation_space.shape[0] * env.observation_space.shape[1], env.action_space.n)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.QNet.to(self.device)
        self.QNet_hat.to(self.device)

    def act(self, state, exploration=False):
        '''Returns the action to take based on the given state'''
        if np.random.rand() < self.epsilon and exploration:
            return self.env.action_space.sample()
        
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            state = state.to(self.device)
            q = self.QNet(state)
            return torch.argmax(q, dim=1).item()

    def save_models(self, episode, path=''):
        '''Saves the models to the given path'''
        torch.save(self.QNet.state_dict(), f'{path}Duel_QNet_model_{episode}.pt')
        torch.save(self.QNet_hat.state_dict(), f'{path}Duel_QNetHat_model_{episode}.pt')

    def load_models(self, first_path, second_path):
        '''Loads the models from the given paths'''
        self.QNet.load_state_dict(torch.load(first_path, map_location=torch.device(self.device)))
        self.QNet_hat.load_state_dict(torch.load(second_path, map_location=torch.device(self.device)))
        self.QNet.eval()
        self.QNet_hat.eval()

    def learn(self, states, actions, rewards, next_states, dones):
        '''Learns from the environment and update the Q network'''
        states = states.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        actions = actions.to(self.device)
        dones = dones.to(self.device)
        for i in range(self.rep):
            indexes = torch.randperm(states.shape[0])[:min(self.batch_size, states.shape[0])].to(self.device)
            self.QNet_optimizer.zero_grad()
            with torch.no_grad():
                best_actions = torch.argmax(self.QNet(next_states[indexes]), dim=1, keepdim=True)
                Q_targets_next = self.QNet_hat(next_states[indexes]).gather(1, best_actions).squeeze(1)
                Q_targets = rewards[indexes].float() + (self.discount * Q_targets_next * torch.logical_not(dones[indexes]))
            Q_expected = self.QNet(states[indexes]).gather(1, actions[indexes].unsqueeze(1)).squeeze(1)
            Q_expected = Q_expected.reshape((-1,))
            loss = F.mse_loss(Q_expected, Q_targets)
            loss.backward()
            self.QNet_optimizer.step()

    def decrease_epsilon(self):
        '''Decreases the epsilon value'''
        self.epsilon = max(self.epsilon * 0.9, 0.05)
        return self.epsilon

    def update_target(self):
        '''Updates the target network'''
        self.QNet_hat.load_state_dict(self.QNet.state_dict())
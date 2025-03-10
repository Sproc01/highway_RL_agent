import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NN_Actor(nn.Module):
    def __init__(self, input_size, output_size):
        super(NN_Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.ac1 = nn.Tanh()
        self.fc2 = nn.Linear(64, 64)
        self.ac2 = nn.Tanh()
        self.fc3 = nn.Linear(64, output_size)
        self.ac3 = nn.Softmax(dim=-1)

    def forward(self, state):
        x = self.fc1(state)
        x = self.ac1(x)
        x = self.fc2(x)
        x = self.ac2(x)
        x = self.fc3(x)
        x = self.ac3(x)
        return x

class NN_Critic(nn.Module):
    def __init__(self, input_size, output_size):
        super(NN_Critic, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.ac1 = nn.Tanh()
        self.fc2 = nn.Linear(64, 64)
        self.ac2 = nn.Tanh()
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, state):
        x = self.fc1(state)
        x = self.ac1(x)
        x = self.fc2(x)
        x = self.ac2(x)
        x = self.fc3(x)
        return x

class Agent_PPO:
    def __init__(self, env, discount=0.9, actor_rep=10, critic_rep=10, clip=0.2, batch_size=100):
        '''Initializes the PPO agent'''
        self.env = env
        self.discount = discount
        self.actor_rep = actor_rep
        self.critic_rep = critic_rep
        self.clip = clip
        self.batch_size = batch_size

        self.actor = NN_Actor(env.observation_space.shape[0] * env.observation_space.shape[1], env.action_space.n)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=5e-4)
        self.critic = NN_Critic(env.observation_space.shape[0] * env.observation_space.shape[1], 1)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=5e-4)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.actor.to(self.device)
        self.critic.to(self.device)

    def mask(self):
        '''Returns a mask for the available actions'''
        mask = torch.ones(self.env.action_space.n, dtype=torch.bool).to(self.device)
        for i in range(self.env.action_space.n):
            if i not in self.env.unwrapped.get_available_actions():
                mask[i] = 0
        return mask

    def act(self, state):
        '''Returns the action to take based on the state'''
        epsilon = 1e-8
        state = state.to(self.device)
        with torch.no_grad():
            probs = self.actor(state)
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                print('Invalid values detected in probabilities output')
            sum_probs = torch.sum(probs)
            if torch.any(torch.isclose(sum_probs, torch.tensor(0.0), atol=1e-8)):
                print('Warning: Sum of probabilities output is close to zero:', sum_probs)
            probs = probs * self.mask()
            sum_probs = torch.sum(probs)
            probs = probs / (sum_probs + epsilon)
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                print('Invalid values detected in probabilities')
                action = self.env.unwrapped.get_available_actions()[0]
            else:
                action = torch.multinomial(probs, 1).item()
            return action

    def save_models(self, episode, path=''):
        '''Saves the models to the path with the episode number'''
        torch.save(self.actor.state_dict(), f'{path}PPO_actor_model_{episode}.pt')
        torch.save(self.critic.state_dict(), f'{path}PPO_critic_model_{episode}.pt')

    def load_models(self, first_path, second_path):
        '''Loads the models from the first_path (actor) and second_path (critic)'''
        self.critic.load_state_dict(torch.load(second_path, weights_only=True))
        self.critic.eval()

        self.actor.load_state_dict(torch.load(first_path, weights_only=True))
        self.actor.eval()

    def learn(self, states, actions, rewards, next_states, dones, masks):
        '''Updates the actor and critic networks'''
        epsilon = 1e-8
        states = states.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        masks = masks.to(self.device)
        with torch.no_grad():
            actions =  F.one_hot(actions.to(torch.int64), num_classes=self.env.action_space.n).to(self.device)
            probs = self.actor(states)
            probs = probs * masks
            probs = probs / (torch.sum(probs, dim = -1, keepdim=True) + epsilon)
            initial_probs = torch.sum(probs * actions, dim=-1, keepdim=True)
            val = self.critic(states)
            new_val = self.critic(next_states)
            future_reward = rewards + self.discount * new_val * torch.logical_not(dones)
            td_err = future_reward - val

        for j in range(self.actor_rep):
            indexes = torch.randperm(states.shape[0])[:min(self.batch_size, states.shape[0])].to(self.device)
            initial_probs = initial_probs[indexes]
            td_err = td_err[indexes]
            states = states[indexes]
            actions = actions[indexes]
            rewards = rewards[indexes]
            next_states = next_states[indexes]
            dones = dones[indexes]
            masks = masks[indexes]
            self.actor_optimizer.zero_grad()
            output = self.actor(states)
            output = output * masks
            output = output / (torch.sum(output, dim = -1, keepdim=True) + epsilon)
            sum_selected = torch.sum(output * actions, dim=-1, keepdim=True)
            sum_initial = torch.sum(initial_probs * actions, dim=-1, keepdim=True)
            imp_s = sum_selected / (sum_initial + epsilon)
            lossActor = torch.min(imp_s * td_err, td_err * torch.clamp(imp_s, 1 - self.clip, 1 + self.clip))
            lossActor = torch.mean(-lossActor)
            lossActor.backward()
            self.actor_optimizer.step()

        for j in range(self.critic_rep):
            indexes = torch.randperm(states.shape[0])[:min(self.batch_size, states.shape[0])].to(self.device)
            states = states[indexes]
            rewards = rewards[indexes]
            next_states = next_states[indexes]
            dones = dones[indexes]
            with torch.no_grad():
                new_val = self.critic(next_states)
            new_val = new_val.reshape(-1)
            future_reward = rewards.float() + self.discount * new_val * torch.logical_not(dones)
            self.critic_optimizer.zero_grad()
            val = self.critic(states)
            val = val.reshape(-1)
            lossCritic = F.mse_loss(val, future_reward)
            lossCritic.backward()
            self.critic_optimizer.step()
    

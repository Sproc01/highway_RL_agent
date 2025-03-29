import gymnasium
import highway_env
import numpy as np
import torch
import random
from agent_PPO import Agent_PPO
import matplotlib.pyplot as plt
from evaluate import evaluate_model

if __name__ == '__main__':
  # Set the seed and create the environment
  np.random.seed(0)
  random.seed(0)
  torch.manual_seed(0)

  # hyperparameters
  batch_size = 24
  clip = 0.2
  critic_rep = 10
  actor_rep = 10
  discount = 0.9

  MAX_EPISODES = 4000
  env_name = 'highway-fast-v0'  # We use the 'fast' env just for faster training, if you want you can use 'highway-v0'

  env = gymnasium.make(env_name,
                      config={'action': {'type': 'DiscreteMetaAction'}, 'duration': 40, 'vehicles_count': 50})

  # Initialize your model
  agent = Agent_PPO(env, discount=discount, actor_rep=actor_rep, critic_rep=critic_rep, clip=clip, batch_size=batch_size)

  state, _ = env.reset()
  state = state.reshape(-1)
  done, truncated = False, False

  episode = 1
  episode_steps = 0
  episode_return = 0

  # for training
  rewards = []
  crashes = []
  states = []
  next_states = []
  actions = []
  masks = []

  # for plotting
  rewards_plot = []
  history = []
  history_ev_ret = []
  history_ev_rew = []

  t = 0
  # Training loop
  while episode <= MAX_EPISODES:
      episode_steps += 1
      t += 1

      # Select the action to be performed by the agent
      mask = agent.mask()
      action = agent.act(torch.tensor(state))
      masks.append(mask)
      actions.append(action)

      next_state, reward, done, truncated, _ = env.step(action)
      next_state = next_state.reshape(-1)

      # Store transition in memory and train your model
      rewards.append(reward)
      rewards_plot.append(reward)
      states.append(state)
      next_states.append(next_state)
      crashes.append(done)

      state = next_state
      episode_return += reward

      if done or truncated:
          # update given the entire trajectory
          agent.learn(torch.from_numpy(np.array(states)),
              torch.from_numpy(np.array(actions)),
              torch.from_numpy(np.array(rewards)),
              torch.from_numpy(np.array(next_states)),
              torch.from_numpy(np.array(crashes)),
              torch.stack(masks))

          if episode % 10 == 0:
            print(f'Total T: {t} Episode Num: {episode} Episode T: {episode_steps} Return: {episode_return:.3f}, truncated: {truncated}')

          # Save training information and model parameters
          history.append(np.mean(rewards_plot))
          rewards_plot = []
          rewards = []
          crashes = []
          states = []
          next_states = []
          actions = []
          masks = []
          if episode % 250 == 0:
            agent.save_models(episode, 'Models/PPO/')
            print('Evaluating model..')
            res = evaluate_model(episode, 'PPO')
            history_ev_ret.append(np.mean(res[0]))
            history_ev_rew.append(np.mean(res[1]))
            print(f'Average reward evaluation: {history_ev_rew[-1]}')
            print(f'Average return evaluation: {history_ev_ret[-1]}')
            

          state, _ = env.reset()
          state = state.reshape(-1)
          episode += 1
          episode_steps = 0
          episode_return = 0
  env.close()

  # Plot the returns
  f = plt.figure()
  plt.plot(history)
  plt.xlabel('Episode')
  plt.ylabel('Average reward')
  plt.title('Average reward over episode')
  f.savefig('Figures/PPO_training.pdf', bbox_inches='tight')

  f = plt.figure()
  plt.plot(history_ev_ret)
  plt.xlabel('Evaluation Step')
  plt.ylabel('Average return')
  plt.title('Average return over steps')
  f.savefig('Figures/PPO_training_eval_return.pdf', bbox_inches='tight')

  f = plt.figure()
  plt.plot(history_ev_rew)
  plt.xlabel('Evaluation Step')
  plt.ylabel('Average reward')
  plt.title('Average reward over steps')
  f.savefig('Figures/PPO_training_eval_avg_reward.pdf', bbox_inches='tight')
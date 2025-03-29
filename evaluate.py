import gymnasium
import highway_env
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from agent_DQN import Agent_DQN
from agent_DuelDQN import Agent_DuelDQN
from agent_PPO import Agent_PPO
from your_baseline import my_baseline


def evaluate_model(episode_load, type_alg, printing=False):
    env_name = 'highway-v0'
    if printing:
        env = gymnasium.make(env_name,
                config={'action': {'type': 'DiscreteMetaAction'}, 'lanes_count': 3, 'ego_spacing': 1.5},
                render_mode='human')
    else:
        env = gymnasium.make(env_name,
                config={'action': {'type': 'DiscreteMetaAction'}, 'lanes_count': 3, 'ego_spacing': 1.5})
    if type_alg == 'DuelDQN':
        agent = Agent_DuelDQN(env=env)
        agent.load_models(f'Models/Duel/Duel_QNet_model_{episode_load}.pt', f'Models/Duel/Duel_QNetHat_model_{episode_load}.pt')
    elif type_alg == 'DQN':
        agent = Agent_DQN(env=env)
        agent.load_models(f'models/DQN/QNet_model_{episode_load}.pt', f'models/DQN/QNetHat_model_{episode_load}.pt')
    elif type_alg == 'PPO':
        agent = Agent_PPO(env=env)
        agent.load_models(f'models/PPO/PPO_actor_model_{episode_load}.pt', f'models/PPO/PPO_critic_model_{episode_load}.pt')
    else:
        raise ValueError('Type of algorithm non supported')
    state, _ = env.reset()
    state = state.reshape(-1)
    done, truncated = False, False
    episode = 1
    episode_steps = 0
    episode_return = 0
    returns = []
    rewards_episode = []
    average_reward = []
    while episode <= 20:
        episode_steps += 1
        action = agent.act(torch.tensor(state))
        state, reward, done, truncated, _ = env.step(action)

        if printing:
            env.render()

        state = state.reshape(-1)
        episode_return += reward
        rewards_episode.append(reward)

        if done or truncated:
            if printing:
                print(f'Episode Num: {episode} Episode T: {episode_steps} Return: {episode_return:.3f}, Crash: {done}')
            
            returns.append(episode_return)
            average_reward.append(np.mean(rewards_episode))
            rewards_episode = []
            
            state, _ = env.reset()
            state = state.reshape(-1)
            episode += 1
            episode_steps = 0
            episode_return = 0

    env.close() 
    return [returns, average_reward]

def evaluate_baseline():
    env_name = 'highway-v0'
    env = gymnasium.make(env_name,
                        config={'action': {'type': 'DiscreteMetaAction'}, 'lanes_count': 3, 'ego_spacing': 1.5},
                        render_mode='human')
    state, _ = env.reset()
    done, truncated = False, False
    episode = 1
    episode_steps = 0
    episode_return = 0

    returns = []
    rewards_episode = []
    average_reward = []

    while episode <= 20:
        episode_steps += 1
        action = my_baseline(env, state)

        state, reward, done, truncated, _ = env.step(action)

        env.render()
        rewards_episode.append(reward)

        episode_return += reward

        if done or truncated:
            print(f'Episode Num: {episode} Episode T: {episode_steps} Return: {episode_return:.3f}, Crash: {done}')

            returns.append(episode_return)
            average_reward.append(np.mean(rewards_episode))
            rewards_episode = []

            state, _ = env.reset()
            episode += 1
            episode_steps = 0
            episode_return = 0

    env.close()
    return [returns, average_reward]


if __name__=='__main__':
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

# ========= RL Agent =========
    algorithm = 'DuelDQN' #'DQN' #'PPO'
    episode = 4000
    printing = True

# ========= Evaluation =========
    print(f'Using the output of {algorithm} after {episode}...')
    res = evaluate_model(episode, algorithm, printing)

    f = plt.figure()
    plt.plot(res[0])
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f'Return over episodes using {algorithm}')
    f.savefig(f'Figures/Return_over_episode_{algorithm}_{episode}.pdf', bbox_inches='tight')

    f = plt.figure()
    plt.plot(res[1])
    plt.xlabel('Episodes')
    plt.ylabel('Average reward')
    plt.title(f'Average reward over episode using {algorithm}')
    f.savefig(f'Figures/Average_reward_over_episode_{algorithm}_{episode}.pdf', bbox_inches='tight')

    plt.show()

# ========= baseline =========
    # print('Using baseline...')

    # res = evaluate_baseline()

    # f = plt.figure()
    # plt.plot(res[0])
    # plt.xlabel('Episodes')
    # plt.ylabel('Returns')
    # plt.title(f'Return over episodes using baseline')
    # f.savefig(f'Figures/Return_over_episode_baseline.pdf', bbox_inches='tight')

    # f = plt.figure()
    # plt.plot(res[1])
    # plt.xlabel('Episodes')
    # plt.ylabel('Average reward')
    # plt.title(f'Average reward over episode using baseline')
    # f.savefig(f'Figures/Average_reward_over_episode_baseline.pdf', bbox_inches='tight')

    # plt.show()


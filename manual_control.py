import gymnasium
import highway_env
import matplotlib.pyplot as plt
import numpy as np

if __name__=='__main__':
    env_name = 'highway-v0'
    env = gymnasium.make(env_name,
                        config={'manual_control': True, 'lanes_count': 3, 'ego_spacing': 1.5},
                        render_mode='human')

    env.reset()
    done, truncated = False, False

    episode = 1
    episode_steps = 0
    episode_return = 0

    returns = []
    rewards_episode = []
    average_reward = []

    print('Manual control starting...')

    while episode <= 20:
        episode_steps += 1
        
        state, reward, done, truncated, _ = env.step(env.action_space.sample())  # With manual control these actions are ignored
        env.render()

        episode_return += reward
        rewards_episode.append(reward)

        if done or truncated:
            print(f'Episode Num: {episode} Episode T: {episode_steps} Return: {episode_return:.3f}, Crash: {done}')

            returns.append(episode_return)
            average_reward.append(np.mean(rewards_episode))
            rewards_episode = []

            env.reset()
            episode += 1
            episode_steps = 0
            episode_return = 0

    env.close()

    # ========= Plotting =========
    f = plt.figure()
    plt.plot(returns)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f'Return over episodes using manual control')
    f.savefig(f'Figures/Return_over_episode_manual.pdf', bbox_inches='tight')

    f = plt.figure()
    plt.plot(average_reward)
    plt.xlabel('Episodes')
    plt.ylabel('Average reward')
    plt.title(f'Average reward over episode using manual control')
    f.savefig(f'Figures/Average_reward_over_episode_manual.pdf', bbox_inches='tight')

    plt.show()

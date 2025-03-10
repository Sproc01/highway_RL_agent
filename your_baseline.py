import numpy as np
import gymnasium
import highway_env

# Actions:
# 0: 'LANE_LEFT',
# 1: 'IDLE',
# 2: 'LANE_RIGHT',
# 3: 'FASTER',
# 4: 'SLOWER'
def my_baseline(env, state):
    '''Baseline agent that drives straight and changes lanes when the ego vehicle is behind another vehicle'''
    epsilon_x = 0.2
    epsilon_y = 0.009
    actions = env.unwrapped.get_available_actions()
    state = state[1:, 1:3]
    state_x = state[:, 0]
    state_y = state[:, 1]
    for i in range(len(state_x)):
        if abs(state_x[i]) < epsilon_x:
            if state_x[i] > 0:
                if abs(state_y[i]) <= epsilon_y:
                    if 0 in actions:
                        return 0
                    else:
                        return 2
                else:
                    return 3
    return 1
        


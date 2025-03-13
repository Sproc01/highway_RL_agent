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
    # Define the epsilon values
    EPSILON_X = 0.2
    EPSILON_Y = 0.009


    actions = env.unwrapped.get_available_actions()
    state = state[1:, 1:3]
    state_x = state[:, 0]
    state_y = state[:, 1]
    for i in range(len(state_x)):
        if abs(state_x[i]) < EPSILON_X: # near along the x axis
            if state_x[i] > 0: # in front of the ego
                if abs(state_y[i]) <= EPSILON_Y: # same in the y axis so it is in front
                    if 0 in actions: # change lane 0 or 2 must be available because 3 lanes
                        return 0
                    else:
                        return 2
                else: # near on x but not on y so free in front
                    return 3
    return 3
        


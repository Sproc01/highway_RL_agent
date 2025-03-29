# Autonomous Driving project

This is the repository for the Autonomous Driving project of the Reinforcement Learning course.

The goal of the agents will be to drive an Autonomous Vehicle through an highway, taking into consideration the presence of other vehicles. For this project it is needed the [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv) library, which can be installed very easily: https://highway-env.farama.org/installation/. 

<img src="https://raw.githubusercontent.com/eleurent/highway-env/gh-media/docs/media/highway_fast_dqn.gif"/>

## Environment specifications

### State space:
The state space consists in a `V x F` array that describes a list of `V = 5` vehicles by a set of features of size 
`F = 5`.

The features for each vehicle are:
- Presence (boolean value)
- Normalized position along the x axis w.r.t. the ego-vehicle
- Normalized position along the y axis w.r.t. the ego-vehicle
- Normalized velocity along the x axis w.r.t. the ego-vehicle
- Normalized velocity along the y axis w.r.t. the ego-vehicle

***Note:*** the first row contains the features of the ego-vehicle, which are the only ones referred to the absolute reference frame.

### Action space
The action space is discrete, and it contains 5 possible actions:
  - Change lane to the left
  - Idle
  - Change lane to the right
  - Go faster
  - Go slower

### Reward function
The reward function is a composition of various terms:
- Bonus term for progressing quickly on the road
- Bonus term for staying on the rightmost lane
- Penalty term for collisions

***Note:*** Look at the documentation for further information and a deeper understanding of the environment: https://highway-env.farama.org/

## Baselines
In this project, the best RL agent is compared against two baselines:
- A baseline defined in order to be an informed guess of the action to be taken.
- The *manual control* policy, in which you will manually control the vehicle using the keyboard. More details on this can be found on the file `manual_control.py` and on the HighwayEnv docs.

## Agents
Three RL agents are defined, for detail look into `agent_DQN.py`, `agent_DuelDQN.py`, `agent_PPO.py`.

## Report
For a detailed description of the project look in TODO

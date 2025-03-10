# Autonomous Driving project

This is the repository for the Autonomous Driving project of the Reinforcement Learning course.

The goal of your agent will be to drive an Autonomous Vehicle through an highway, taking into consideration the presence of other vehicles. For this project you will use the [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv) library, which can be installed very easily: https://highway-env.farama.org/installation/. 

<img src="https://raw.githubusercontent.com/eleurent/highway-env/gh-media/docs/media/highway_fast_dqn.gif"/>

Recall that you can choose to implement the Deep-RL algorithm that you prefer. Moreover, the provided code is just a skeleton to help you get started, feel free to modify it. Finally, don't worry if the collision rate at test time is not exactly zero.

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

***Note:*** you are encouraged to take a look at the documentation for further information and a deeper understanding of the environment: https://highway-env.farama.org/

## Baselines
As written on the instructions for the projects, you have to implement a baseline policy to be compared against the trained RL agent. For this project, you will need to compare the performances of your agent against two baselines:
- The one you define
- The *manual control* policy, in which you will manually control the vehicle using the keyboard. More details on this can be found on the file `manual_control.py` and on the HighwayEnv docs (the only code you have to add here is the one needed to save information that you want to include in the report).

***Note:*** It may not be super easy to outperform the manual control policy with your RL agent. This is not a problem, the goal is to obtain comparable results.

## Bonus
As written on the instructions for the projects, you may play around with the problem definition, for example:
- Use multiple algorithms
- Consider a different state representation
- Modify the reward function
- Change some environment configurations (refer again at the [documentation](https://highway-env.farama.org/))

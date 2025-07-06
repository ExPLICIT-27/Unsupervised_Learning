import numpy as np
from state_action_utils import *

num_states = 6
num_actions = 2


terminal_left_reward = 100
terminal_right_reward = 40
each_step_reward = 0

#discount factor
gamma = 0.5

misstep_prob = 0 # > 0 for a stochastic environment
generate_visualization(terminal_left_reward, terminal_right_reward, each_step_reward, gamma, misstep_prob)

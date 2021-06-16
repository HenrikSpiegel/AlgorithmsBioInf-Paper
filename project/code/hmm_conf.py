# Config file
import numpy as np

states = 2 #redundant 
state_names=["Fair", "Loaded"]

#Give matrix in the same order regarding states

#NOTE must be in real space - for viberti the probabilities,
# will be transformed to log space

initial_prob = [1.0/states, 1.0/states]

transition_matrix = np.array([
    [0.95,0.05],
    [0.1, 0.9]
])

symbols = "123456"

emission_probs = np.array([
    [1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6],
    [1.0/10, 1.0/10, 1.0/10, 1.0/10, 1.0/10, 5.0/10]
])

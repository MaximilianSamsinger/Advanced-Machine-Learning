import gym
import numpy as np
import matplotlib.pyplot as plt

NUM_RUNS = 50

MAX_EPISODE_STEPS = 5000
NUM_EPISODES = 100
NUM_ACTIONS = 3

env = gym.make('MountainCar-v0').env
env._max_episode_steps = MAX_EPISODE_STEPS

POLYNOMIAL_FEATURES = True
POLYNOMIAL_DEGREE = 2



'''
https://github.com/openai/gym/wiki/MountainCar-v0
Summary for me:
    There are 3 different actions (push left, nothing, push right) which
    correspond to (0, 1, 2).
    An state is a pair (position,velocity).
    Position ranges from -1.2 to 0.6 and velocity from -0.07 to 0.07
'''

def convert_to_features(state, action, polynomial = POLYNOMIAL_FEATURES,
                        max_degree = POLYNOMIAL_DEGREE):
    
    if polynomial:
        '''
        Converts the state to a polynomial of degree max_degree.
        However, we cap the degree of action at 1, since we cannot expect
        higher powers of action to deliver meaningful extra features.
        (Action takes values in (0, 1, 2) )
        '''     
        
        
        subfeatures = []
        for n in range(max_degree + 1):
            for k in range(n+1):
                subfeatures += [state[0]**(n-k) * state[1]**(k)]
        
        subfeatures = np.array(subfeatures)
        features = np.zeros(2*len(subfeatures))
        features[:len(subfeatures)] = subfeatures
        features[len(subfeatures):] = action * subfeatures
        
        return features
    
    
    else:
        ''' Directly returns the state and action as a feature '''
        
        return np.append(state, action)

    

state = env.reset()
    
NUM_FEATURES = len(convert_to_features(state, 0))

learning_rate = 1e-4
discount = 0.9

def action_value(state, action, weights):
    ''' Approximate the action value of an state'''
    features = convert_to_features(state, action)
    return features @ weights

def choose_action(state, weights, epsilon = 0.33):
    ''' A random action will be picked with probability epsilon '''
    if epsilon > np.random.uniform():
        action = np.random.randint(0, NUM_ACTIONS)
    else:
        action = np.argmax([action_value(state, action, weights) 
                    for action in range(NUM_ACTIONS)])  
    return action

def update_weights(previous_state, state, action, next_action,
                   weights, reward, done = False):      
    gradient = convert_to_features(previous_state, action)
    if done:
        update = learning_rate * (reward 
                        - action_value(previous_state, action, weights))
        
    else:
        update = learning_rate * (reward 
                        + discount * action_value(state, next_action, weights)
                        - action_value(previous_state, action, weights))
    update *= gradient
    weights -= update # Gradient descent
    return weights

''' Calculate average reward per run and per episode '''
average_rewards = np.zeros((NUM_RUNS, NUM_EPISODES))


for j in range(NUM_RUNS):
    ''' Start of another run '''
    print('Run:', j)
    weights = np.random.rand(NUM_FEATURES)
    
    for k in range(NUM_EPISODES):
        ''' Start of another episode '''
        state = env.reset()
        action = choose_action(state, weights.copy())
        cumulative_reward = 0
        for step in range(MAX_EPISODE_STEPS):
            #env.render()
            previous_state = state
            state, reward, done, info = env.step(action)
            next_action = choose_action(state, weights.copy())
            weights = update_weights(previous_state, state, action, 
                                       next_action, weights.copy(), reward, done)
            
            cumulative_reward += reward
            if done:
                break
            action = next_action
        average_reward = cumulative_reward
        average_rewards[j,k] = average_reward

average_rewards_per_run = average_rewards.sum(axis=0)/NUM_RUNS/MAX_EPISODE_STEPS

plt.plot(average_rewards_per_run)
plt.xlabel('Episode')
plt.ylabel('Average reward per episode and run')
if POLYNOMIAL_FEATURES:
    title = 'Using polynomial features\n'
else:
    title = 'Using states and actions directly as features\n'
title += 'Average of ' + str(NUM_RUNS) + ' runs with ' \
    + str(MAX_EPISODE_STEPS) + ' steps per episode'
plt.title(title)
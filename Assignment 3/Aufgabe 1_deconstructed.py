import gym
import numpy as np

env = gym.make('MountainCar-v0')

goal_steps = 200
score_requirement = -198
intial_games = 10000


'''
https://github.com/openai/gym/wiki/MountainCar-v0
Summary for me:
    There are 3 different actions (push left, nothing, push right) which
    correspond to (0, 1, 2).
    An state is a pair (position,velocity).
    Position ranges from -1.2 to 0.6 and velocity from -0.07 to 0.07
'''
NUM_ACTIONS = 3

def convert_to_features(state, action):
    return np.append(state, action)

'''
def convert_to_features(state, action, max_degree = 2):
    return np.append(state,action)
    
    subfeatures = []
    for n in range(1,max_degree + 1):
        for k in range(n+1):
            subfeatures += [state[0]**(n-k) * state[1]**(k)]
    
    features = np.zeros((NUM_ACTIONS,len(subfeatures)))
    features[action] = subfeatures
    features = np.array(features).flatten()
    print(features)
    return np.append(features,np.array([action,1]))
'''    

state = env.reset()
    
NUM_FEATURES = len(convert_to_features(state, 0))

learning_rate = 1e-2
discount = 0.9
weights = np.zeros(NUM_FEATURES)

def action_value(state, action, weights):
    ''' Approximate the action value of an state'''
    features = convert_to_features(state, action)
    return features @ weights

def choose_action(state, weights, epsilon = 0.02):
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
                        + discount 
                        * action_value(state, next_action, weights)
                        - action_value(previous_state, action, weights))
    else:
        update = learning_rate * (reward 
                    - action_value(previous_state, action, weights))
    update *= gradient
    weights += update
    return weights


for k in range(1, 10001):
    if k%10 == 0:
        print('Lap:',k)

    state = env.reset()
    action = choose_action(state, weights)
    for _ in range(goal_steps):
        if k%100 == 0:
            env.render()
        
        previous_state = state
        state, reward, done, info = env.step(action)
        next_action = choose_action(state, weights)
        weights = update_weights(previous_state, state, action, 
                                   next_action, weights, reward, done)
        if done:
            break
        action = next_action

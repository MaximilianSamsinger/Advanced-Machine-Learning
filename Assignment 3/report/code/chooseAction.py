def choose_action(state, weights, epsilon = 0.02):
    ''' A random action will be picked with probability epsilon '''
    if epsilon > np.random.uniform():
        action = np.random.randint(0, NUM_ACTIONS)
    else:
        action = np.argmax([action_value(state, action, weights) 
                    for action in range(NUM_ACTIONS)])  
    return action
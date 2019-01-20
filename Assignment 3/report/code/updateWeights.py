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
    weights -= update
    return weights
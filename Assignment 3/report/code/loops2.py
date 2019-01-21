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
            env.render()
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
for j in range(NUM_RUNS):
    ''' Start of another run '''
    print('Run:', j)
    weights = np.random.randn(NUM_FEATURES)
    
    for k in range(NUM_EPISODES):
        ''' Start of another episode '''
        state = env.reset()
        action = choose_action(state, weights)
        cumulative_reward = 0
        for step in range(MAX_EPISODE_STEPS):
            if k % 100 == 102:
                env.render()
            
            previous_state = state
            state, reward, done, info = env.step(action)
            next_action = choose_action(state, weights)
            weights = update_weights(previous_state, state, action, 
                                       next_action, weights, reward, done)
            
            cumulative_reward += reward
            if done:
                break
            action = next_action
        average_reward = cumulative_reward / (step + 1)
        average_rewards[j,k] = average_reward
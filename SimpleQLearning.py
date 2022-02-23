import numpy as np
import gym
import random
import time
import matplotlib.pyplot as plt

env = gym.make("FrozenLake-v1")

# env = [[0, 0, 0, 0],
#        [0, 0, 0, 0],
#        [0, 0, 0, 0],
#        [0, 0, 0, 0]]
ACTION_SPACE = ('U', 'D', 'L', 'R')


class Env:
    def __init__(self, rows, cols, start):
        self.rows = rows
        self.cols = cols
        self.i = start[0]
        self.j = start[1]
        self.goal_i = 3
        self.goal_j = 3

    def reset(self):
        self.i = 0
        self.j = 0
        return self.rows * self.i + self.j

    def step(self, action):
        done = False
        reward = 0
        if action == 0:
            if self.j > 0:
                self.j -= 1
        elif action == 1:
            if self.j < self.cols - 1:
                self.j += 1
        elif action == 2:
            if self.i > 0:
                self.i -= 1
        elif action == 3:
            if self.i < self.rows - 1:
                self.i += 1
        if self.i == self.goal_i and self.j == self.goal_j:
            done = True
            reward = 1

        if (self.i, self.j) == (2, 2) or (self.i, self.j) == (1, 2):
            reward = -1
            done = True

        return (self.rows * self.i + self.j, reward, done, None)

    def sample(self):
        return random.randint(0, 3)

    def world(self, pos=(0, 0)):
        world = np.zeros((self.rows, self.cols))
        world[2, 2] = -1
        world[1, 2] = -1
        world[3, 3] = 1
        world[pos[0], pos[1]] = 2
        return world


env = Env(4, 4, (0, 0))

action_size = 4
state_size = 16

qtable = np.zeros((state_size, action_size))
print(qtable)

total_episodes = 20000  # Total episodes
learning_rate = 0.8  # Learning rate
max_steps = 99  # Max steps per episode
gamma = 0.95  # Discounting rate

# Exploration parameters
epsilon = 1.0  # Exploration rate
max_epsilon = 1.0  # Exploration probability at start
min_epsilon = 0.01  # Minimum exploration probability
decay_rate = 0.01  # Exponential decay rate for exploration prob

# List of rewards
rewards = []

# 2 For life or until learning is stopped
for episode in range(total_episodes):
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0

    for step in range(max_steps):
        # 3. Choose an action a in the current world state (s)
        ## First we randomize a number
        exp_exp_tradeoff = random.uniform(0, 1)

        ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state, :])

        # Else doing a random choice --> exploration
        else:
            action = env.sample()

            # Take the action (a) and observe the outcome state(s') and reward (r)
            new_state, reward, done, info = env.step(action)

            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            # qtable[new_state,:] : all the actions we can take from new state
            qtable[state, action] = qtable[state, action] + learning_rate * (
                    reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])

            total_rewards += reward
            # Our new state is state
            state = new_state

            # If done (if we're dead) : finish episode
            if done == True:
                break

        episode += 1
        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        rewards.append(total_rewards)
        # print(total_rewards)
        # if episode % 1000 == 0:
        #     print(qtable)

print("Score over time: " + str(sum(rewards) / total_episodes))
print(qtable)

env.reset()

for episode in range(1):
    state = env.reset()
    step = 0
    done = False
    print("****************************************************")
    print("EPISODE ", episode)
    print(env.world((env.i, env.j)))

    for step in range(max_steps):
        # env.render()
        print()
        time.sleep(0.5)
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(qtable[state, :])

        new_state, reward, done, info = env.step(action)
        print(env.world((env.i, env.j)))

        if done:
            break
        state = new_state
# env.close()

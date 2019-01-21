import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


from scores.score_logger import ScoreLogger

class DefaultParams:
    def __init__(self):
        self.ENV_NAME = "CartPole-v1"
        self.GAMMA = 0.95
        self.LEARNING_RATE = 0.001

        self.MEMORY_SIZE = 1000000
        self.BATCH_SIZE = 20

        self.EXPLORATION_MAX = 1.0
        self.EXPLORATION_MIN = 0.01
        self.EXPLORATION_DECAY = 0.995


class DQNSolver:

    def __init__(self, observation_space, action_space, params):
        self.params = params
        self.exploration_rate = self.params.EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=self.params.MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=self.params.LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < self.params.BATCH_SIZE:
            return
        batch = random.sample(self.memory, self.params.BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + self.params.GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= self.params.EXPLORATION_DECAY
        self.exploration_rate = max(self.params.EXPLORATION_MIN, self.exploration_rate)


def cartpole(iteration = 0, params = None):
    if params is None:
        params = DefaultParams()
    env = gym.make(params.ENV_NAME)
    score_logger = ScoreLogger(params.ENV_NAME, iteration)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space, params)
    run = 0
    done = False
    while done == False:
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        while True:
            step += 1
            #env.render()
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                done = score_logger.add_score(step, run)
                break
            dqn_solver.experience_replay()


if __name__ == "__main__":
    cartpole()

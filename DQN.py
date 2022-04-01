import numpy as np
import random
import gym
import time
import signal
import tensorflow as tf
from collections import deque
from tensorflow.keras.layers import Input, MaxPooling2D, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard


# (Hyper)parameters
LR = 0.001
EPSILON = 0.1
GAMMA = 0.95

REPLAY_CAPACITY = 2000
TRAINING_LIMITATION = 1000
SET_WEIGHTS_FREQENCY = 50
BATCH_SIZE = 32

EPISODES = 1000
ENVNAME = 'CartPole-v1'

GAME_ENV = None



class Network():
    def __init__(self, N_STATES, N_ACTIONS) -> None:
        self.N_STATES = N_STATES
        self.N_ACTIONS = N_ACTIONS
    def Qnet_FC(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.N_STATES, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.N_ACTIONS, activation='linear', kernel_initializer='he_uniform'))
        model.compile(loss='mse', optimizer=Adam(lr=LR), metrics=['mse'])
        return model

    def Qnet(self):
        X_inputs = Input(shape=(self.N_STATES,))
        # 1st
        X = Conv2D(16, 3, activation='relu')(X_inputs)
        X = MaxPooling2D((2, 2))(X)
        # 2rd
        X = Conv2D(32, 3, activation='relu')(X)
        X = MaxPooling2D((2, 2))(X)
        X = Dropout(0.25)(X)
        X = Flatten()(X)

        X = Dense(32, activation = 'relu')(X)
        X = Dropout(0.25)(X)
        X = Dense(self.N_ACTIONS, activation = 'linear')(X)

        model = Model(inputs = X_inputs, outputs = X)
        return model
    
class DQNAgent():
    def __init__(self, N_STATES, N_ACTIONS):
        nn = Network(N_STATES, N_ACTIONS)
        self.original_model = nn.Qnet_FC()
        self.target_model = nn.Qnet_FC()
        self.replay_buffer = deque(maxlen=REPLAY_CAPACITY)
        self.training_counts = 0
        self.N_STATES = N_STATES
        self.N_ACTIONS = N_ACTIONS
    
    def actEgreedy(self, S):
        S = np.reshape(S, (1, self.N_STATES))
        if np.random.rand() >= EPSILON: # Exploit
            Q = self.original_model.predict(S)
            A = np.argmax(Q)
        else:
            A = np.random.randint(0, self.N_ACTIONS) # Explore
        return A
    
    def appendExperienceReplay(self, S, A, R, S_next, done):
        transition = (S, A, R, S_next, done)
        self.replay_buffer.append(transition)
    
    def sampleData(self):
        return random.sample(self.replay_buffer, BATCH_SIZE)
    
    def updateTargetModel(self):
        print('Update target model')
        self.target_model.set_weights(self.original_model.get_weights())
    
    def training(self):
        if self.training_counts % SET_WEIGHTS_FREQENCY == 0:
            self.updateTargetModel()
        self.training_counts += 1
        batch_data = self.sampleData()
        S_batch, A_batch, R_batch, S_next_batch, terminal_batch = [], [], [], [], []
        for data in batch_data:
            S_batch.append(data[0])
            A_batch.append(data[1])
            R_batch.append(data[2])
            S_next_batch.append(data[3])
            terminal_batch.append(data[4])
        
        S_batch = np.array(S_batch)
        S_next_batch = np.array(S_next_batch)
        
        Q_batch = self.original_model(S_batch)
        Q_target_batch = np.array(Q_batch, copy=True)
        Q_next_batch = self.target_model(S_next_batch)
        for i in range(BATCH_SIZE):
            terminal = terminal_batch[i]
            Q_target = R_batch[i] if terminal else R_batch[i] + GAMMA * np.max(Q_next_batch, axis=-1)[i]
            Q_target_batch[i][A_batch[i]] = Q_target
        result = self.original_model.fit(x=S_batch, y=Q_target_batch, verbose=0)
        # return result.history['loss']

# def handler(signum, frame):
#     GAME_ENV.close()
#     exit(1)
        
def main():
    ENV = gym.make(ENVNAME)  # make game env
    global GAME_ENV
    GAME_ENV = ENV
    N_STATES = ENV.observation_space.shape[0] # 4
    N_ACTIONS = ENV.action_space.n # 2
    Dqn_agent = DQNAgent(N_STATES, N_ACTIONS)
    step_counter_list = []
    scores = []

    for episode in range(EPISODES):
        S = ENV.reset()
        step_counts = 0
        score = 0
        # start_time = time.time()
        while True:
            # if step_counter == 100:
            #     end_time = time.time()
            #     print('100 step: ', end_time - start_time)
            # ENV.render()
            A = Dqn_agent.actEgreedy(S)
            S_next, R, terminal, _ = ENV.step(A) # perform an action
            R = -R if terminal else R # If fail, suffer punishment
            Dqn_agent.appendExperienceReplay(S, A, R, S_next, terminal) # Experience replay
            current_buffer_size = len(Dqn_agent.replay_buffer)
            if current_buffer_size > BATCH_SIZE:
                Dqn_agent.training() # training if volume is greater than batch size.
            
            S = S_next
            score += R
            step_counts += 1
            
            if terminal:
                # Dqn_agent.target_model.set_weights(Dqn_agent.original_model.get_weights()) # Update target model every episode
                step_counter_list.append(step_counts)
                break
                # net.plot(net.ax, step_counter_list)
            
        scores.append(score)
        print("Episode: {}, Total reward: {}, Total step: {}".format(episode, score, step_counter_list[-1]))
    ENV.close()
    print('Scores: ', scores)
    print('Steps: ', step_counter_list)


if __name__ == '__main__':
    a = time.time()
    main()
    b = time.time()
    print('total time', b - a)
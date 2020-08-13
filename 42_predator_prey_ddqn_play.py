import sys
import pylab
import random
import numpy as np
import os
import time, datetime
from collections import deque
from keras.layers import *
from keras.models import Sequential,Model
import keras
from keras import backend as K_back
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D, MaxPooling2D

state_size = 64
action_size = 5

model_path = "save_model/"
graph_path = "save_graph/"

if not os.path.isdir(model_path):
    os.mkdir(model_path)

if not os.path.isdir(graph_path):
    os.mkdir(graph_path)
    
load_model = True

class DDQN_agent:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        # get size of state and action
        self.progress = " "
        self.action_size = action_size
        self.state_size = state_size
        
        self.episode = 0
        
        # These are hyper parameters for the DDQN_agent
        self.learning_rate = 0.001
        self.discount_factor = 0.99
                
        self.hidden1, self.hidden2 = 251, 251
        
        self.ep_trial_step = 1000
        
        # Parameter for Experience Replay
        self.size_replay_memory = 10000
        self.batch_size = 32
        
        self.input_shape = (9,9,1)
        
        # Experience Replay 
        self.memory = deque(maxlen=self.size_replay_memory)
        
        # create main model and target model
        self.model = self.build_model()

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),activation='relu', \
                         padding = 'valid', input_shape=self.input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu', padding = 'valid'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        # model.add(Dense(self.hidden1, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.hidden2, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        q_value = self.model.predict(state)
        return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        #in every action put in the memory
        self.memory.append((state, action, reward, next_state, done))
                        
def main():
    
    # Agents generation
    agnt_0 = DDQN_agent(state_size, action_size)
    agnt_1 = DDQN_agent(state_size, action_size)
    agnt_2 = DDQN_agent(state_size, action_size)
    agnt_3 = DDQN_agent(state_size, action_size)
    
    if load_model:
        agnt_0.model.load_weights(model_path + "/Preda_prey_ddqn_0.h5")
        agnt_1.model.load_weights(model_path + "/Preda_prey_ddqn_1.h5")
        agnt_2.model.load_weights(model_path + "/Preda_prey_ddqn_2.h5")
        agnt_3.model.load_weights(model_path + "/Preda_prey_ddqn_3.h5")
    display_time = datetime.datetime.now()
    print("\n\n Game start at :",display_time)
    
    agnt_0.episode = 0
    time_step = 0
    
    while agnt_0.episode < 20:
        
        done = False
        ep_step = 0
        
        n_rows = 11
        n_cols = 11

        state_prey = np.zeros((9,9))
        state_predators = np.zeros((9,9))

        prey_row = 0
        prey_col = 0
        predator_rows = np.zeros((1,4))
        predator_cols = np.zeros((1,4))
        
        """
        len_unique = 0
        while len_unique != 5:
            init_state = np.random.randint(low=0, high=64, size=5)
            len_unique = len(np.unique(init_state))
                        
        row_idx, col_idx = divmod(init_state[0],11)
        """
        row_idx, col_idx = 4,4 
        state_prey[row_idx][col_idx] = 1
        prey_row = row_idx
        prey_col = col_idx
        
        """
        for idx in range(4):
            row_idx,col_idx = divmod(init_state[idx+1],11)
            predator_rows[0][idx] = row_idx
            predator_cols[0][idx] = col_idx
        """
        
        game_prey = np.zeros((n_rows,n_cols))
        game_prey[1:10,1:10] = state_prey
        
        predator_0 = np.zeros((n_rows,n_cols))
        predator_1 = np.zeros((n_rows,n_cols))
        predator_2 = np.zeros((n_rows,n_cols))
        predator_3 = np.zeros((n_rows,n_cols))
 
        prey_row += 1
        prey_col += 1
        predator_rows = [[1,1,9,9]]
        predator_cols = [[1,9,1,9]]
        
        predator_0[int(predator_rows[0][0])][int(predator_cols[0][0])] = 2
        predator_1[int(predator_rows[0][1])][int(predator_cols[0][1])] = 2
        predator_2[int(predator_rows[0][2])][int(predator_cols[0][2])] = 2
        predator_3[int(predator_rows[0][3])][int(predator_cols[0][3])] = 2

        game_arr_frame = np.full((n_rows, n_cols), 8)
        game_arr_frame[1:10,1:10] = np.zeros((9,9))
        game_arr = game_arr_frame + game_prey + predator_0 + predator_1 + predator_2 + predator_3
        
        act_arr = np.zeros((1,4))
        
        while not done and ep_step < agnt_0.ep_trial_step:

            ep_step += 1
            time_step += 1
            
            # prey action define
            prey_action = random.randrange(5)
            if prey_action == 0:
                if game_arr[prey_row+1][prey_col] == 0:
                    prey_row += 1
            if prey_action == 1:
                if game_arr[prey_row-1][prey_col] == 0:
                    prey_row -= 1
            if prey_action == 2:
                if game_arr[prey_row][prey_col-1] == 0:
                    prey_col -= 1
            if prey_action == 3:
                if game_arr[prey_row][prey_col+1] == 0:
                    prey_col += 1
            # print("Prey Action :",prey_action)
            
            game_prey = np.zeros((11,11))
            game_prey[prey_row][prey_col] = 1
            game_arr = game_arr_frame + game_prey + predator_0 + predator_1 + predator_2 + predator_3
            # print(game_arr)
            
            state_t = game_arr[1:10,1:10]
            state = copy.deepcopy(state_t)
            state = state.reshape(1,9,9,1)
        
            # agent action define
            agnt_0_row = int(predator_rows[0][0])
            agnt_1_row = int(predator_rows[0][1])
            agnt_2_row = int(predator_rows[0][2])
            agnt_3_row = int(predator_rows[0][3])

            agnt_0_col = int(predator_cols[0][0])
            agnt_1_col = int(predator_cols[0][1])
            agnt_2_col = int(predator_cols[0][2])
            agnt_3_col = int(predator_cols[0][3])
            
            act_arr[0][0] = agnt_0.get_action(state)
            act_arr[0][1] = agnt_1.get_action(state)
            act_arr[0][2] = agnt_2.get_action(state)
            act_arr[0][3] = agnt_3.get_action(state)
            
            # Assume that below action is correct
            for idx in range(4):
                if act_arr[0][idx] == 0:
                    if game_arr[int(predator_rows[0][idx]+1)][int(predator_cols[0][idx])] == 0:
                        predator_rows[0][idx] += 1
                if act_arr[0][idx] == 1:
                    if game_arr[int(predator_rows[0][idx]-1)][int(predator_cols[0][idx])] == 0:
                        predator_rows[0][idx] -= 1
                if act_arr[0][idx] == 2:
                    if game_arr[int(predator_rows[0][idx])][int(predator_cols[0][idx]-1)] == 0:
                        predator_cols[0][idx] -= 1
                if act_arr[0][idx] == 3:
                    if game_arr[int(predator_rows[0][idx])][int(predator_cols[0][idx]+1)] == 0:
                        predator_cols[0][idx] += 1
                        
                                                
            predator_0 = np.zeros((11,11))
            predator_1 = np.zeros((11,11))
            predator_2 = np.zeros((11,11))
            predator_3 = np.zeros((11,11))            
            predator_0[int(predator_rows[0][0])][int(predator_cols[0][0])] = 2
            predator_1[int(predator_rows[0][1])][int(predator_cols[0][1])] = 2
            predator_2[int(predator_rows[0][2])][int(predator_cols[0][2])] = 2
            predator_3[int(predator_rows[0][3])][int(predator_cols[0][3])] = 2
            
            game_arr = game_arr_frame + game_prey + predator_0 + predator_1 + predator_2 + predator_3
            
            next_state_t = game_arr[1:10,1:10]
            next_state = copy.deepcopy(next_state_t)
            next_state = next_state.reshape(1,9,9,1)
        
            distances = np.zeros((1,4))
            
            for idx in range(4):
                temp_dis = np.abs(predator_rows[0][idx] - prey_row) \
                        + np.abs(prey_col-predator_cols[0][idx])
                distances[0][idx] = temp_dis
            
            if game_arr[prey_row+1][prey_col] != 0:
                if game_arr[prey_row-1][prey_col] != 0:
                    if game_arr[prey_row][prey_col-1] != 0:
                        if game_arr[prey_row][prey_col+1] != 0:
                            done = True                
            
            if done:
                reward_0 = 0
                reward_1 = 0
                reward_2 = 0
                reward_3 = 0
            else:
                reward_0 = -1 - np.sum(distances) - distances[0][0] + 4
                reward_1 = -1 - np.sum(distances) - distances[0][1] + 4
                reward_2 = -1 - np.sum(distances) - distances[0][2] + 4
                reward_3 = -1 - np.sum(distances) - distances[0][3] + 4
            
            agnt_0.append_sample(state, int(act_arr[0][0]), reward_0, next_state, done)
            agnt_1.append_sample(state, int(act_arr[0][1]), reward_1, next_state, done)
            agnt_2.append_sample(state, int(act_arr[0][2]), reward_2, next_state, done)
            agnt_3.append_sample(state, int(act_arr[0][3]), reward_3, next_state, done)
            
            state = next_state
            
            if done or ep_step == 500:
                print(game_arr)
                agnt_0.episode += 1
                print("episode :{:>5d} / ep_step :{:>5d} ".format(agnt_0.episode, ep_step))
                break
    sys.exit()
                    
if __name__ == "__main__":
    main()

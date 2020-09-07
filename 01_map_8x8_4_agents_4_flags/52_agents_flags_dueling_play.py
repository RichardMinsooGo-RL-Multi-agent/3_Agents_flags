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
n_ticks = 8

class Agents_Flags:
    
    def __init__(self):
        self.n_ticks = n_ticks
        
    def reset_env(self):

        self.n_rows = self.n_ticks+2
        self.n_cols = self.n_ticks+2

        state_flags  = np.zeros((n_ticks,n_ticks), dtype=int)
        state_agents = np.zeros((n_ticks,n_ticks), dtype=int)

        self.flag_rows    = np.zeros((1,4), dtype=int)[0]
        self.flag_cols    = np.zeros((1,4), dtype=int)[0]
        self.agent_rows   = np.zeros((1,4), dtype=int)[0]
        self.agent_cols   = np.zeros((1,4), dtype=int)[0]
        
        """
        len_unique = 0
        while len_unique != n_ticks:
            init_state = np.random.randint(low=0, high=64, size=n_ticks)
            len_unique = len(np.unique(init_state))

        for idx in range(4):
            row_idx,col_idx = divmod(init_state[idx],n_ticks)
            state_flags[row_idx][col_idx] = 1
            self.flag_rows[idx] = row_idx
            self.flag_cols[idx] = col_idx

        for idx in range(4):
            row_idx,col_idx = divmod(init_state[idx+4],n_ticks)
            self.agent_rows[idx] = row_idx
            self.agent_cols[idx] = col_idx

        self.game_flags = np.zeros((self.n_rows,n_ticks+2))
        self.game_flags[1:n_ticks+1,1:n_ticks+1] = state_flags
        
        # print(self.flag_rows)
        # print(self.flag_cols)
        # print(self.agent_rows)
        # print(self.agent_cols)

        self.flag_rows += 1
        self.flag_cols += 1
        self.agent_rows += 1
        self.agent_cols += 1
        """        
        
        self.flag_rows  = [1,n_ticks,1,n_ticks]
        self.flag_cols  = [1,1,n_ticks,n_ticks]
        self.agent_rows = [4,4,5,5]
        self.agent_cols = [4,5,4,5]
        self.game_flags = np.zeros((self.n_rows,self.n_cols))
        
        self.game_flags[self.flag_rows[0]][self.flag_cols[0]] = 1
        self.game_flags[self.flag_rows[1]][self.flag_cols[1]] = 1
        self.game_flags[self.flag_rows[2]][self.flag_cols[2]] = 1
        self.game_flags[self.flag_rows[3]][self.flag_cols[3]] = 1
        
        self.agent_0 = np.zeros((self.n_rows,self.n_cols), dtype=int)
        self.agent_1 = np.zeros((self.n_rows,self.n_cols), dtype=int)
        self.agent_2 = np.zeros((self.n_rows,self.n_cols), dtype=int)
        self.agent_3 = np.zeros((self.n_rows,self.n_cols), dtype=int)
        
        self.agent_0[self.agent_rows[0]][self.agent_cols[0]] = 2
        self.agent_1[self.agent_rows[1]][self.agent_cols[1]] = 2
        self.agent_2[self.agent_rows[2]][self.agent_cols[2]] = 2
        self.agent_3[self.agent_rows[3]][self.agent_cols[3]] = 2

        self.game_arr_frame = np.full((self.n_rows, self.n_cols), 8, dtype=int)
        self.game_arr_frame[1:n_ticks+1,1:n_ticks+1] = np.zeros((n_ticks,n_ticks), dtype=int)
        
        # print(self.agent_0.shape)
        # print(self.agent_1.shape)
        # print(self.agent_2.shape)
        # print(self.agent_3.shape)
        
        self.game_arr = self.game_arr_frame + self.game_flags + self.agent_0 + self.agent_1 + self.agent_2 + self.agent_3
        
        # print(self.game_arr.astype(int))
        
        return self.game_arr
        
    def p0_frame_step(self, action):
        
        if self.game_arr[self.agent_rows[0]][self.agent_cols[0]] == 3:
            action = 4

        if action == 0:
            if self.game_arr[self.agent_rows[0]+1][self.agent_cols[0]] < 2:
                self.agent_rows[0] += 1
        if action == 1:
            if self.game_arr[self.agent_rows[0]-1][self.agent_cols[0]] < 2:
                self.agent_rows[0] -= 1
        if action == 2:
            if self.game_arr[self.agent_rows[0]][self.agent_cols[0]-1] < 2:
                self.agent_cols[0] -= 1
        if action == 3:
            if self.game_arr[self.agent_rows[0]][self.agent_cols[0]+1] < 2:
                self.agent_cols[0] += 1

        self.agent_0 = np.zeros((n_ticks+2,n_ticks+2))
        self.agent_0[self.agent_rows[0]][self.agent_cols[0]] = 2

        self.game_arr = self.game_arr_frame + self.game_flags + self.agent_0 + self.agent_1 + self.agent_2 + self.agent_3
        
        return self.game_arr
    
    def p1_frame_step(self, action):
        
        if self.game_arr[self.agent_rows[1]][self.agent_cols[1]] == 3:
            action = 4

        if action == 0:
            if self.game_arr[self.agent_rows[1]+1][self.agent_cols[1]] < 2:
                self.agent_rows[1] += 1
        if action == 1:
            if self.game_arr[self.agent_rows[1]-1][self.agent_cols[1]] < 2:
                self.agent_rows[1] -= 1
        if action == 2:
            if self.game_arr[self.agent_rows[1]][self.agent_cols[1]-1] < 2:
                self.agent_cols[1] -= 1
        if action == 3:
            if self.game_arr[self.agent_rows[1]][self.agent_cols[1]+1] < 2:
                self.agent_cols[1] += 1

        self.agent_1 = np.zeros((n_ticks+2,n_ticks+2))
        self.agent_1[self.agent_rows[1]][self.agent_cols[1]] = 2
        self.game_arr = self.game_arr_frame + self.game_flags + self.agent_0 + self.agent_1 + self.agent_2 + self.agent_3

        return self.game_arr
    
    def p2_frame_step(self, action):
        
        if self.game_arr[self.agent_rows[2]][self.agent_cols[2]] == 3:
            action = 4

        if action == 0:
            if self.game_arr[self.agent_rows[2]+1][self.agent_cols[2]] < 2:
                self.agent_rows[2] += 1
        if action == 1:
            if self.game_arr[self.agent_rows[2]-1][self.agent_cols[2]] < 2:
                self.agent_rows[2] -= 1
        if action == 2:
            if self.game_arr[self.agent_rows[2]][self.agent_cols[2]-1] < 2:
                self.agent_cols[2] -= 1
        if action == 3:
            if self.game_arr[self.agent_rows[2]][self.agent_cols[2]+1] < 2:
                self.agent_cols[2] += 1

        self.agent_2 = np.zeros((n_ticks+2,n_ticks+2))
        self.agent_2[self.agent_rows[2]][self.agent_cols[2]] = 2

        self.game_arr = self.game_arr_frame + self.game_flags + self.agent_0 + self.agent_1 + self.agent_2 + self.agent_3

        return self.game_arr
    
    
    def p3_frame_step(self, action):
        
        done = False
        
        if self.game_arr[self.agent_rows[3]][self.agent_cols[3]] == 3:
            action = 4
            
        if action == 0:
            if self.game_arr[self.agent_rows[3]+1][self.agent_cols[3]] < 2:
                self.agent_rows[3] += 1
        if action == 1:
            if self.game_arr[self.agent_rows[3]-1][self.agent_cols[3]] < 2:
                self.agent_rows[3] -= 1
        if action == 2:
            if self.game_arr[self.agent_rows[3]][self.agent_cols[3]-1] < 2:
                self.agent_cols[3] -= 1
        if action == 3:
            if self.game_arr[self.agent_rows[3]][self.agent_cols[3]+1] < 2:
                self.agent_cols[3] += 1

        self.agent_3 = np.zeros((n_ticks+2,n_ticks+2))
        self.agent_3[self.agent_rows[3]][self.agent_cols[3]] = 2

        self.game_arr = self.game_arr_frame + self.game_flags + self.agent_0 + self.agent_1 + self.agent_2 + self.agent_3
        
        distance_0 = np.zeros((1,4))[0]
        distance_1 = np.zeros((1,4))[0]
        distance_2 = np.zeros((1,4))[0]
        distance_3 = np.zeros((1,4))[0]

        for idx in range(4):
            if self.game_arr[self.flag_rows[idx]][self.flag_cols[idx]] == 1:
                distance_0[idx] = np.abs(self.flag_rows[idx]-self.agent_rows[0]) \
                    + np.abs(self.flag_cols[idx]-self.agent_cols[0])
            else:
                distance_0[idx] = self.n_rows + self.n_cols

        for idx in range(4):
            if self.game_arr[self.flag_rows[idx]][self.flag_cols[idx]] == 1:
                distance_1[idx] = np.abs(self.flag_rows[idx]-self.agent_rows[1]) \
                    + np.abs(self.flag_cols[idx]-self.agent_cols[1])
            else:
                distance_1[idx] = self.n_rows + self.n_cols

        for idx in range(4):
            if self.game_arr[self.flag_rows[idx]][self.flag_cols[idx]] == 1:
                distance_2[idx] = np.abs(self.flag_rows[idx]-self.agent_rows[2]) \
                    + np.abs(self.flag_cols[idx]-self.agent_cols[2])
            else:
                distance_2[idx] = self.n_rows + self.n_cols

        for idx in range(4):
            if self.game_arr[self.flag_rows[idx]][self.flag_cols[idx]] == 1:
                distance_3[idx] = np.abs(self.flag_rows[idx]-self.agent_rows[3]) \
                    + np.abs(self.flag_cols[idx]-self.agent_cols[3])
            else:
                distance_3[idx] = self.n_rows + self.n_cols

        dist_fl_0 = np.zeros((1,4))[0]
        dist_fl_1 = np.zeros((1,4))[0]
        dist_fl_2 = np.zeros((1,4))[0]
        dist_fl_3 = np.zeros((1,4))[0]

        for idx in range(4):
            temp_dis = np.abs(self.flag_rows[0]-self.agent_rows[idx]) \
                    + np.abs(self.flag_cols[0]-self.agent_cols[idx])
            dist_fl_0[idx] = temp_dis

        for idx in range(4):
            temp_dis = np.abs(self.flag_rows[1]-self.agent_rows[idx]) \
                    + np.abs(self.flag_cols[1]-self.agent_cols[idx])
            dist_fl_1[idx] = temp_dis

        for idx in range(4):
            temp_dis = np.abs(self.flag_rows[2]-self.agent_rows[idx]) \
                    + np.abs(self.flag_cols[2]-self.agent_cols[idx])
            dist_fl_2[idx] = temp_dis

        for idx in range(4):
            temp_dis = np.abs(self.flag_rows[3]-self.agent_rows[idx]) \
                    + np.abs(self.flag_cols[3]-self.agent_cols[idx])
            dist_fl_3[idx] = temp_dis

        game_dist = np.min(dist_fl_0) + np.min(dist_fl_1) + np.min(dist_fl_2) + np.min(dist_fl_3)

        remain_flags   = np.count_nonzero(self.game_arr == 1)
        self.flag_n_agent   = np.count_nonzero(self.game_arr == 3)
        
        if self.flag_n_agent == 4:
            done = True

        reward_arr = np.zeros((1,4))[0]
        if done:
            reward_arr = reward_arr
        else:
            reward_arr[0] = -1 - np.min(distance_0) - remain_flags - game_dist
            reward_arr[1] = -1 - np.min(distance_1) - remain_flags - game_dist
            reward_arr[2] = -1 - np.min(distance_2) - remain_flags - game_dist
            reward_arr[3] = -1 - np.min(distance_3) - remain_flags - game_dist
                
        return self.game_arr, reward_arr, done
    
class DQN_agnt_0:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        # get size of state and action
        self.progress = " "
        self.action_size = action_size
        self.state_size = state_size
        
        self.episode = 0
        
        # These are hyper parameters for the DQN_agnt_0
        self.learning_rate = 0.0005
        
        self.hidden1, self.hidden2 = 251, 251
        
        self.ep_trial_step = 1000
        
        # Parameter for Experience Replay
        self.size_replay_memory = 10000
        self.batch_size = 32
        
        self.input_shape = (n_ticks,n_ticks,1)
        
        # Experience Replay 
        self.memory = deque(maxlen=self.size_replay_memory)
        
        # create main model and target model
        self.model = self.build_model()

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        
        state = Input(shape=self.input_shape)        
        
        net1 = Convolution2D(32, kernel_size=(3, 3),activation='relu', \
                             padding = 'valid', input_shape=self.input_shape)(state)
        net2 = Convolution2D(64, kernel_size=(3, 3), activation='relu', padding = 'valid')(net1)
        net3 = MaxPooling2D(pool_size=(2, 2))(net2)
        net4 = Flatten()(net3)
        lay_2 = Dense(units=self.hidden2,activation='relu',kernel_initializer='he_uniform',\
                  name='hidden_layer_1')(net4)
        value_= Dense(units=1,activation='linear',kernel_initializer='he_uniform',\
                      name='Value_func')(lay_2)
        ac_activation = Dense(units=self.action_size,activation='linear',\
                              kernel_initializer='he_uniform',name='action')(lay_2)
        
        #Compute average of advantage function
        avg_ac_activation = Lambda(lambda x: K_back.mean(x,axis=1,keepdims=True))(ac_activation)
        
        #Concatenate value function to add it to the advantage function
        concat_value = Concatenate(axis=-1,name='concat_0')([value_,value_])
        concat_avg_ac = Concatenate(axis=-1,name='concat_ac_{}'.format(0))([avg_ac_activation,avg_ac_activation])

        for i in range(1,self.action_size-1):
            concat_value = Concatenate(axis=-1,name='concat_{}'.format(i))([concat_value,value_])
            concat_avg_ac = Concatenate(axis=-1,name='concat_ac_{}'.format(i))([concat_avg_ac,avg_ac_activation])

        #Subtract concatenated average advantage tensor with original advantage function
        ac_activation = Subtract()([ac_activation,concat_avg_ac])
        
        #Add the two (Value Function and modified advantage function)
        merged_layers = Add(name='final_layer')([concat_value,ac_activation])
        model = Model(inputs = state,outputs=merged_layers)
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
    
    # DQN_agnt_0 에이전트의 생성
    agnt_0 = DQN_agnt_0(state_size, action_size)
    agnt_1 = DQN_agnt_0(state_size, action_size)
    agnt_2 = DQN_agnt_0(state_size, action_size)
    agnt_3 = DQN_agnt_0(state_size, action_size)
    
    game = Agents_Flags()
    
    if load_model:
        agnt_0.model.load_weights(model_path + "/Model_dueling_0.h5")
        agnt_1.model.load_weights(model_path + "/Model_dueling_1.h5")
        agnt_2.model.load_weights(model_path + "/Model_dueling_2.h5")
        agnt_3.model.load_weights(model_path + "/Model_dueling_3.h5")

    last_n_game_score = deque(maxlen=20)
    last_n_game_score.append(agnt_0.ep_trial_step)
    avg_ep_step = np.mean(last_n_game_score)
    
    display_time = datetime.datetime.now()
    print("\n\n Game start at :",display_time)
    
    start_time = time.time()
    agnt_0.episode = 0
    time_step = 0
    
    while agnt_0.episode < 5:
        
        # reset environment
        game.game_arr = game.reset_env()
        
        done = False
        ep_step = 0
        
        act_arr = np.zeros((1,4), dtype=int )[0]
        
        state_t = game.game_arr[1:n_ticks+1,1:n_ticks+1]
        state = copy.deepcopy(state_t)
        state = state.reshape(1,n_ticks,n_ticks,1)
        
        # while ep_step < 20:
        while not done and ep_step < agnt_0.ep_trial_step:

            ep_step += 1
            time_step += 1
            
            act_arr[0] = agnt_0.get_action(state)
            game_arr = game.p0_frame_step(act_arr[0])
            
            next_state_t = game_arr[1:n_ticks+1,1:n_ticks+1]
            next_state = copy.deepcopy(next_state_t)
            next_state = next_state.reshape(1,n_ticks,n_ticks,1)
            state_1 = next_state
            
            act_arr[1] = agnt_1.get_action(state_1)
            game_arr = game.p1_frame_step(act_arr[1])
            
            next_state_t = game_arr[1:n_ticks+1,1:n_ticks+1]
            next_state = copy.deepcopy(next_state_t)
            next_state = next_state.reshape(1,n_ticks,n_ticks,1)
            state_2 = next_state
            
            act_arr[2] = agnt_2.get_action(state_2)
            game_arr = game.p2_frame_step(act_arr[2])
            
            next_state_t = game_arr[1:n_ticks+1,1:n_ticks+1]
            next_state = copy.deepcopy(next_state_t)
            next_state = next_state.reshape(1,n_ticks,n_ticks,1)
            state_3 = next_state
            
            act_arr[3] = agnt_3.get_action(state_3)
            game_arr, reward_arr, done = game.p3_frame_step(act_arr[3])
            
            next_state_t = game_arr[1:n_ticks+1,1:n_ticks+1]
            next_state = copy.deepcopy(next_state_t)
            next_state = next_state.reshape(1,n_ticks,n_ticks,1)
            
            """
            agnt_0.append_sample(state, act_arr[0], reward_arr[0], state_1, done)
            agnt_1.append_sample(state_1, act_arr[1], reward_arr[1], state_2, done)
            agnt_2.append_sample(state_2, act_arr[2], reward_arr[2], state_3, done)
            agnt_3.append_sample(state_3, act_arr[3], reward_arr[3], next_state, done)
            """
            # if self.flag_n_agent > 3:
            #     print(game_arr)
            if ep_step % 500 == 0:
                print("\nfound flags :",game.flag_n_agent)
                print(game_arr[1:n_ticks+1,1:n_ticks+1].astype(int))
            
            state = next_state
            
            if done or ep_step == agnt_0.ep_trial_step:
                agnt_0.episode += 1
                last_n_game_score.append(ep_step)
                avg_ep_step = np.mean(last_n_game_score)
                print("found flags are 4 at time step :", time_step)
                #print("episode finish!\n",game_arr)
                print("episode :{:>5d} / ep_step :{:>5d} / last 20 game avg :{:>4.1f}".format(agnt_0.episode, ep_step, avg_ep_step))
                break
                
    sys.exit()
                    
if __name__ == "__main__":
    main()

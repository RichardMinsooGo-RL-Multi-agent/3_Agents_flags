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
        
        # train time define
        self.training_time = 10*60
        
        self.episode = 0
        
        # These are hyper parameters for the DQN_agnt_0
        self.learning_rate = 0.0005
        self.discount_factor = 0.99
        
        self.epsilon_max = 0.149
        self.epsilon_min = 0.0005
        self.epsilon_decay = 0.99
        self.epsilon_rate = self.epsilon_max
        
        self.hidden1, self.hidden2 = 251, 251
        
        self.ep_trial_step = 1000
        
        # Parameter for Experience Replay
        self.size_replay_memory = 10000
        self.batch_size = 32
        
        self.input_shape = (n_ticks,n_ticks,1)
        
        # Experience Replay 
        self.memory = deque(maxlen=self.size_replay_memory)
        
        # Parameter for Target Network
        self.target_update_cycle = 100

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.Copy_Weights()

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

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        
        minibatch = random.sample(self.memory, self.batch_size)

        states      = np.zeros((self.batch_size, n_ticks, n_ticks, 1))
        next_states = np.zeros((self.batch_size, n_ticks, n_ticks, 1))
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            states[i]      = minibatch[i][0]
            actions.append(  minibatch[i][1])
            rewards.append(  minibatch[i][2])
            next_states[i] = minibatch[i][3]
            dones.append(    minibatch[i][4])

        q_value          = self.model.predict(states)
        q_value_next     = self.model.predict(next_states)
        tgt_q_value_next = self.target_model.predict(next_states)

        for i in range(self.batch_size):
            # like Q Learning, get maximum Q value at s'
            # But from target model
            if dones[i]:
                q_value[i][actions[i]] = rewards[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                a = np.argmax(q_value_next[i])
                q_value[i][actions[i]] = rewards[i] + self.discount_factor * (tgt_q_value_next[i][a])

        # make minibatch which includes target q value and predicted q value
        # and do the model fit!
        self.model.fit(states, q_value, batch_size=self.batch_size, epochs=1, verbose=0)
        
        if self.epsilon_rate > self.epsilon_min:
            self.epsilon_rate *= self.epsilon_decay
                        
    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        #Exploration vs Exploitation
        if np.random.rand() <= self.epsilon_rate:
            # print("Random action selected!!")
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        #in every action put in the memory
        self.memory.append((state, action, reward, next_state, done))
    
    # after some time interval update the target model to be same with model
    def Copy_Weights(self):
        self.target_model.set_weights(self.model.get_weights())

def main():
    
    # DQN_agnt_0 에이전트의 생성
    agnt_0 = DQN_agnt_0(state_size, action_size)
    agnt_1 = DQN_agnt_0(state_size, action_size)
    agnt_2 = DQN_agnt_0(state_size, action_size)
    agnt_3 = DQN_agnt_0(state_size, action_size)
    
    game = Agents_Flags()
    
    if load_model:
        agnt_0.model.load_weights(model_path + "/Model_ddqn_0.h5")
        agnt_1.model.load_weights(model_path + "/Model_ddqn_1.h5")
        agnt_2.model.load_weights(model_path + "/Model_ddqn_2.h5")
        agnt_3.model.load_weights(model_path + "/Model_ddqn_3.h5")

    last_n_game_score = deque(maxlen=20)
    last_n_game_score.append(agnt_0.ep_trial_step)
    avg_ep_step = np.mean(last_n_game_score)
    
    display_time = datetime.datetime.now()
    print("\n\n Game start at :",display_time)
    
    start_time = time.time()
    agnt_0.episode = 0
    time_step = 0
    
    # while agnt_0.episode < 50:
    
    while time.time() - start_time < agnt_0.training_time and avg_ep_step > 30:
        
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
            if len(agnt_0.memory) < agnt_0.size_replay_memory:
                agnt_0.progress = "Exploration"
                agnt_1.progress = "Exploration"
                agnt_2.progress = "Exploration"
                agnt_3.progress = "Exploration"
                
            else :
                agnt_0.progress = "Training"
                agnt_1.progress = "Training"
                agnt_2.progress = "Training"
                agnt_3.progress = "Training"

            ep_step += 1
            time_step += 1
            
            act_arr[0] = agnt_0.get_action(state)
            game_arr = game.p0_frame_step(act_arr[0])
            
            next_state_t = game_arr[1:n_ticks+1,1:n_ticks+1]
            next_state = copy.deepcopy(next_state_t)
            next_state = next_state.reshape(1,n_ticks,n_ticks,1)
            state_1 = next_state
            
            # print(act_arr[0])
            # print(game_arr.astype(int))
            
            act_arr[1] = agnt_1.get_action(state_1)
            game_arr = game.p1_frame_step(act_arr[1])
            
            next_state_t = game_arr[1:n_ticks+1,1:n_ticks+1]
            next_state = copy.deepcopy(next_state_t)
            next_state = next_state.reshape(1,n_ticks,n_ticks,1)
            state_2 = next_state
            
            # print(act_arr[1])
            # print(game_arr.astype(int))
            
            act_arr[2] = agnt_2.get_action(state_2)
            game_arr = game.p2_frame_step(act_arr[2])
            
            next_state_t = game_arr[1:n_ticks+1,1:n_ticks+1]
            next_state = copy.deepcopy(next_state_t)
            next_state = next_state.reshape(1,n_ticks,n_ticks,1)
            state_3 = next_state
            
            # print(act_arr[2])
            # print(game_arr.astype(int))
            
            
            act_arr[3] = agnt_3.get_action(state_3)
            game_arr, reward_arr, done = game.p3_frame_step(act_arr[3])
            
            next_state_t = game_arr[1:n_ticks+1,1:n_ticks+1]
            next_state = copy.deepcopy(next_state_t)
            next_state = next_state.reshape(1,n_ticks,n_ticks,1)
        
            # print(act_arr[3])
            # print(game_arr.astype(int))
            # print(reward_arr.astype(int))
            # sys.exit()
            
            agnt_0.append_sample(state, act_arr[0], reward_arr[0], state_1, done)
            agnt_1.append_sample(state_1, act_arr[1], reward_arr[1], state_2, done)
            agnt_2.append_sample(state_2, act_arr[2], reward_arr[2], state_3, done)
            agnt_3.append_sample(state_3, act_arr[3], reward_arr[3], next_state, done)
            
            # if self.flag_n_agent > 3:
            #     print(game_arr)
            if ep_step % 500 == 0:
                print("\nfound flags :",game.flag_n_agent)
                print(game_arr[1:n_ticks+1,1:n_ticks+1].astype(int))
            
            state = next_state
            
            if agnt_0.progress == "Training":
                agnt_0.train_model()
                if done or ep_step % agnt_0.target_update_cycle == 0:
                    # return# copy q_net --> target_net
                    agnt_0.Copy_Weights()

            if agnt_1.progress == "Training":
                agnt_1.train_model()
                if done or ep_step % agnt_1.target_update_cycle == 0:
                    agnt_1.Copy_Weights()
                    
            if agnt_2.progress == "Training":
                agnt_2.train_model()
                if done or ep_step % agnt_2.target_update_cycle == 0:
                    agnt_2.Copy_Weights()
                    
            if agnt_3.progress == "Training":
                agnt_3.train_model()
                if done or ep_step % agnt_3.target_update_cycle == 0:
                    agnt_3.Copy_Weights()
                    
            if done or ep_step == agnt_0.ep_trial_step:
                if agnt_0.progress == "Training":
                    # print(game_arr)
                    agnt_0.episode += 1
                    last_n_game_score.append(ep_step)
                    avg_ep_step = np.mean(last_n_game_score)
                print("found flags are 4 at time step :", time_step)
                #print("episode finish!\n",game_arr)
                print("episode :{:>5d} / ep_step :{:>5d} / last 20 game avg :{:>4.1f}".format(agnt_0.episode, ep_step, avg_ep_step))
                print("\n")
                break
                
    agnt_0.model.save_weights(model_path + "/Model_ddqn_0.h5")
    agnt_1.model.save_weights(model_path + "/Model_ddqn_1.h5")
    agnt_2.model.save_weights(model_path + "/Model_ddqn_2.h5")
    agnt_3.model.save_weights(model_path + "/Model_ddqn_3.h5")
    
    e = int(time.time() - start_time)
    print(' Elasped time :{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
    sys.exit()
                    
if __name__ == "__main__":
    main()

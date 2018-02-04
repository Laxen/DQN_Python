# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 16:38:28 2016

@author: alex
"""

import tensorflow as tf
from collections import deque
import random
import numpy as np
import scipy.misc as sp
import gym
import matplotlib.pyplot as plt
import os
import psutil
from bitarray import bitarray
import time

np.set_printoptions(threshold=np.nan)

NUM_ACTIONS = 4
REPLAY_MEMORY_LENGTH = 1000000 # Length of replay memory
#OBSERVE_TIME = 50000 # Timesteps before training
OBSERVE_TIME = 50 # Timesteps before training
BATCH_SIZE = 32 # Size of minibatch to sample from replay memory
GAMMA = 0.95 # Discount factor
UPDATE_TIME = 10000 # Time between net copy
NUM_EPISODES = 10000000
VISUALIZE_EPISODES = 20
#VISUALIZE_EPISODES = 1
SAVE_TIME = 10000 # Time between network weight saves

INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.1
EPSILON = INITIAL_EPSILON
#EPSILON = 0
EPSILON_DECAY_TIME = 1000000

NUM_FRAMES = 4

MAXIMUM_MEMORY = 7000 # Maximum usage of RAM (in MB)

class State:
    def __init__(self, num_frames):
        self.num_frames = num_frames
        self.state = np.zeros([84,84,num_frames],dtype=np.uint8)
        
    def add(self, frame):
        frame_pp = self.preprocess(frame)
        
        for i in reversed(range(self.num_frames-1)):
            self.state[:,:,i+1] = self.state[:,:,i]
        self.state[:,:,0] = frame_pp
            
        return np.copy(self.state) # Return a COPY of the array, otherwise all states will be the same array
        
    def preprocess(self, frame):
        """ Preprocesses a frame and returns the result """
        
        # Extract luminance
        #lum = np.sum(frame, axis=2)
        #lum = np.divide(lum, 3)
        lum = frame[:,:,0]
        
        
        # Returns preprocessed frame        
        pp = sp.imresize(lum, [110,84])  # Removes score and lives  
        pp = np.uint8((pp > 0)*255)
        return pp[26:110,:]
        
    def reset(self):
        """ Resets the state """
        self.state = np.zeros([84,84,self.num_frames],dtype=np.uint8)

class Q_Net:
    def __init__(self, num_actions):        
        # Initialize parameters
        self.time_step = 0
        self.num_actions = num_actions
        self.RMS_learning_rate = 0.00025
        self.RMS_decay = 0.99
        self.RMS_momentum = 0.0
        
        # Initialize replay memory
        self.replay_memory = deque()
        
        # Reset the graph and create TensorFlow session 
        tf.reset_default_graph()
        self.session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        
        # Initialize Q and target Q-network
        self.state_input,self.Q_values,self.W_conv_1,self.b_conv_1,self.W_conv_2,self.b_conv_2,self.W_conv_3,self.b_conv_3,self.W_fc_1,self.b_fc_1,self.W_fc_2,self.b_fc_2 = self.build()
        self.state_input_T,self.Q_values_T,self.W_conv_1_T,self.b_conv_1_T,self.W_conv_2_T,self.b_conv_2_T,self.W_conv_3_T,self.b_conv_3_T,self.W_fc_1_T,self.b_fc_1_T,self.W_fc_2_T,self.b_fc_2_T = self.build()
        
        self.copy_Q_net_op = [self.W_conv_1_T.assign(self.W_conv_1),self.b_conv_1_T.assign(self.b_conv_1),self.W_conv_2_T.assign(self.W_conv_2),self.b_conv_2_T.assign(self.b_conv_2),self.W_conv_3_T.assign(self.W_conv_3),self.b_conv_3_T.assign(self.b_conv_3),self.W_fc_1_T.assign(self.W_fc_1),self.b_fc_1_T.assign(self.b_fc_1),self.W_fc_2_T.assign(self.W_fc_2),self.b_fc_2_T.assign(self.b_fc_2)]
    
        self.create_training_method()
        
        # Initialize TensorFlow session and saving/loading
        self.saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        self.session.run(init)
        
        print("Checking for saved network...")
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            print("Loading saved network...")
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Successfully loaded: " + str(checkpoint.model_checkpoint_path))
        else:
            print("Could not find saved network weights")
        
    def build(self):
        state_input = tf.placeholder(tf.float32, [None, 84, 84, 4])
        
        # Convolution layer 1
        W_conv_1 = self.weight_variable([8,8,4,32])
        b_conv_1 = self.bias_variable([32])
        conv_1_strides = [1,4,4,1]
        conv_1 = tf.nn.relu(tf.nn.conv2d(state_input, W_conv_1, strides=conv_1_strides, padding="VALID") + b_conv_1)     
        
        # Convolution layer 2
        W_conv_2 = self.weight_variable([4,4,32,64])
        b_conv_2 = self.bias_variable([64])
        conv_2_strides = [1,2,2,1]
        conv_2 = tf.nn.relu(tf.nn.conv2d(conv_1, W_conv_2, strides=conv_2_strides, padding="VALID") + b_conv_2)
        
        # Convolution layer 3
        W_conv_3 = self.weight_variable([3,3,64,64])
        b_conv_3 = self.bias_variable([64])
        conv_3_strides = [1,1,1,1]
        conv_3 = tf.nn.relu(tf.nn.conv2d(conv_2, W_conv_3, strides=conv_3_strides, padding="VALID") + b_conv_3)
        
        # Fully connected with ReLU
        W_fc_1 = self.weight_variable([7*7*64, 512])
        b_fc_1 = self.bias_variable([512])
        conv_3_flat = tf.reshape(conv_3, [-1, 7*7*64])
        fc_1 = tf.nn.relu(tf.matmul(conv_3_flat, W_fc_1) + b_fc_1)
        
        # Fully connected
        W_fc_2 = self.weight_variable([512,self.num_actions])
        b_fc_2 = self.bias_variable([self.num_actions])
        fc_2 = tf.matmul(fc_1, W_fc_2) + b_fc_2
        
        Q_values = fc_2
        
        return state_input,Q_values,W_conv_1,b_conv_1,W_conv_2,b_conv_2,W_conv_3,b_conv_3,W_fc_1,b_fc_1,W_fc_2,b_fc_2
    
    def copy_Q_net(self):
        print("COPYING Q_NET")
        self.session.run(self.copy_Q_net_op)
    
    def create_training_method(self):
        self.action = tf.placeholder(tf.float32, [None, self.num_actions]) # One-hot vector
        Q_current = tf.reduce_sum(tf.mul(self.Q_values, self.action), axis=1) # NOTE: elementwise multiplication
        self.Q_target = tf.placeholder(tf.float32, [None])
        
        # Loss
        self.loss = tf.reduce_mean(tf.square(self.Q_target - Q_current))
        self.train_step = tf.train.RMSPropOptimizer(learning_rate=self.RMS_learning_rate, decay=self.RMS_decay, momentum=self.RMS_momentum, epsilon=1e-6).minimize(self.loss)

    def train_network(self):
        # Obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_memory, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]
        
        state_batch_arr = np.zeros([32,84,84,4])
        next_state_batch_arr = np.zeros([32,84,84,4])
        for i in range(BATCH_SIZE):
            temp = np.fromstring(state_batch[i].unpack(), dtype=np.uint8)
            state_batch_arr[i,:,:,:] = np.reshape(temp, [84,84,4])
            
            temp = np.fromstring(next_state_batch[i].unpack(), dtype=np.uint8)
            next_state_batch_arr[i,:,:,:] = np.reshape(temp, [84,84,4])
            
        state_batch = state_batch_arr
        next_state_batch = next_state_batch_arr
        
        
        #print("length: " + str(state_batch[0][:,:,0].shape)
#        plt.figure(1)
#        for i in range(4):
#            plt.gray()
#            plt.subplot(1,4,i+1)
#            plt.imshow(state_batch_arr[0][:,:,i])
#            
#        plt.figure(2)
#        for i in range(4):
#            plt.gray()
#            plt.subplot(1,4,i+1)
#            plt.imshow(state_batch_arr[1][:,:,i])
#        
#        raw_input("done")
        
        # Calculate Q-targets
        Q_target_batch = []
        Q_value_batch = self.session.run(self.Q_values_T, feed_dict={self.state_input_T: next_state_batch})
        for i in range(0, BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                Q_target_batch.append(reward_batch[i])
            else:
                Q_target_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))
                
        # Train network with Q-targets
        self.session.run(self.train_step, feed_dict={
            self.Q_target: Q_target_batch,
            self.action: action_batch,
            self.state_input: state_batch})
            
        if self.time_step % SAVE_TIME == 0:
            print("Saving network")
            self.saver.save(self.session, "./saved_networks/" + "network" + "-dqn", global_step = self.time_step)
            print("Network saved!")
            
        # Copy net if needed
        if self.time_step % UPDATE_TIME == 0:
            self.copy_Q_net()
        
    def add_to_replay_memory(self, state, action, reward, next_state, terminal):
        """ Adds the sequence to replay memory and trains the network """
        """ Action needs to be one-hot """
        
        # Convert state to bits to save memory        
        state_bin = state > 0
        state_bits = bitarray()
        state_bits.pack(state_bin.tostring())
        
        next_state_bin = next_state > 0
        next_state_bits = bitarray()
        next_state_bits.pack(next_state_bin.tostring())
        
        #self.replay_memory.append((state, action, reward, next_state, terminal))
        self.replay_memory.append((state_bits, action, reward, next_state_bits, terminal))
        
        if len(self.replay_memory) > REPLAY_MEMORY_LENGTH:
            self.replay_memory.popleft()
            
        if self.time_step > OBSERVE_TIME:
            self.train_network()
            
        self.time_step += 1
    
    def get_action(self, state):
        """ Returns simple action """
        """ Also returns Q-value for the state-action pair for debugging purposes """
        
        # Expand dimensions to simulate a batch of 1
        state = np.expand_dims(state, axis=0)
        
        Q_values = self.session.run(self.Q_values, feed_dict={self.state_input: state})
        action = np.argmax(Q_values)
        
        return action, Q_values[0,action]
        
        # HERE I SKIP FRAME_PER_ACTION THINGY
        # I ALSO DO NOT UPDATE EPSILON HERE, THAT SHOULD BE DONE OUTSIDE OF DQN        
    
    def get_time_step(self):
        return self.time_step
    
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)
        
    def get_replay_memory(self):
        return self.replay_memory
        
env = gym.make("Breakout-v0")
state_keeper = State(NUM_FRAMES)
q_net = Q_Net(NUM_ACTIONS)
time_step = 0
Q_value = 0
process = psutil.Process(os.getpid())

# Used for clearing the terminal
clear = "\n" * 40

start_time = 0
end_time = 0

for i in range(NUM_EPISODES):  
    mem = process.memory_info().rss / 1000000.0 # Memory usage in MB
    rep_mem_len = len(q_net.get_replay_memory()) # Replay memory length
    
    print(clear)
    print("EPISODE: " + str(i) + ", TIMESTEP: " + str(time_step) + ", Q-value sample: " + str(Q_value) + ", EPSILON: " + str(EPSILON))
    print("VISUALIZATION IN " + str(VISUALIZE_EPISODES - i % VISUALIZE_EPISODES) + " EPISODES")
    print("NETWORK SAVING IN " + str(SAVE_TIME - time_step % SAVE_TIME) + " TIME STEPS")
    print("REPLAY MEMORY SIZE " + str(rep_mem_len) + ", (" + str(round(rep_mem_len/np.double(REPLAY_MEMORY_LENGTH)*100,2)) + "%)")
    print("TOTAL MEMORY USAGE " + str(round(mem,2)) + ", (" + str(round(mem/MAXIMUM_MEMORY*100,2)) + "%)")
    print("TIME ELAPSED DURING EPISODE: " + str(round(end_time - start_time,2)))
    
    if process.memory_info().rss / 1000000.0 > MAXIMUM_MEMORY:
        print("\nWARNING: MEMORY USAGE HIGHER THAN " + str(MAXIMUM_MEMORY) + " MB")
        mem_incr = input("Type number of MB to increase memory limit with: ")
        MAXIMUM_MEMORY += mem_incr
    
    start_time = time.time()
    
    s = env.reset()
    
    state_keeper.reset()
    state = state_keeper.add(s)
    
    while 1:        
        if i % VISUALIZE_EPISODES == 0:
            env.render()
        
        # Epsilon-greedy action selection
        # NOTE: Q_value only used for debugging purposes
        if np.random.rand() < EPSILON:
            action = np.random.randint(NUM_ACTIONS)
        else:
            action, Q_value = q_net.get_action(state)
            
        # Make one-hot action
        action_onehot = np.zeros(NUM_ACTIONS)
        action_onehot[action] = 1            
        
        # Decay epsilon if we are done observing
        time_step = q_net.get_time_step()
        if EPSILON > FINAL_EPSILON and time_step > OBSERVE_TIME:
            EPSILON -= (INITIAL_EPSILON - FINAL_EPSILON) / EPSILON_DECAY_TIME
       
        s,reward,terminal,_ = env.step(action)
        next_state = state_keeper.add(s)        
        
        # Add sequence to replay memory AND train network
        q_net.add_to_replay_memory(state, action_onehot, reward, next_state, terminal)
        
        state = next_state
        
        if terminal:
            break
        
    end_time = time.time()
        
        
        
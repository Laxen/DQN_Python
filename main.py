# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 17:12:20 2016

@author: alex
"""

import tensorflow as tf
import gym
import numpy as np
import scipy.misc as sp
import matplotlib.pyplot as plt


class Sequence:
    """ Handles the sequence structure """
    
    def __init__(self, num_frames):
        self.num_frames = num_frames
        #self.seq = np.zeros([num_frames, frame_length])
        self.seq = np.zeros([84,84,num_frames])
        self.index = 0
    
    def add(self, frame):
        """ Preprocesses frame and shifts it in from left in sequence """
        """ Returns sequence """
        
        # Preprocess frame and stack
        frame_pp = self.preprocess(frame)
        
        # Shift in frame in seq
        for i in reversed(range(self.num_frames-1)):
            self.seq[:,:,i+1] = self.seq[:,:,i]
        self.seq[:,:,0] = frame_pp
        
        return self.seq
    
    def preprocess(self, frame):
        """ Preprocesses a frame and returns the result """
        
        # Extract luminance
        lum = np.sum(frame, axis=2)
        lum = np.divide(lum, 3)
        
        # Returns preprocessed frame
        return sp.imresize(lum, [84,84]) 
        
    def get_sequence(self):
        """ Returns the entire preprocessed sequence as 84x84x4 """
        
        return self.seq
        
        
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

#class NN_Basics:
#    @staticmethod
#    def weight_variable(shape):
#        initial = tf.truncated_normal(shape, stddev=0.1)
#        return tf.Variable(initial)
#        
#    @staticmethod
#    def bias_variable(shape):
#        initial = tf.constant(0.1, shape=shape)
#        return tf.Variable(initial)
    
class Q_Net:
    """ Using NHWC """
    
    def __init__(self):
        # Initialize parameters
        self.time_step = 0
        
        # Initialize Q and target Q netowork
        self.state_input,self.Q,self.W_conv_1,self.b_conv_1,self.W_conv_2,self.b_conv_2,self.W_conv_3,self.b_conv_3,self.W_fc_1,self.b_fc_1,self.W_fc_2,self.b_fc_2 = self.build()
        self.state_input_T,self.Q_T,self.W_conv_1_T,self.b_conv_1_T,self.W_conv_2_T,self.b_conv_2_T,self.W_conv_3_T,self.b_conv_3_T,self.W_fc_1_T,self.b_fc_1_T,self.W_fc_2_T,self.b_fc_2_T = self.build()
        
        self.copy_Q_net_op = [self.W_conv_1_T.assign(self.W_conv_1),self.b_conv_1_T.assign(self.b_conv_1),self.W_conv_2_T.assign(self.W_conv_2),self.b_conv_2_T.assign(self.b_conv_2),self.W_conv_3_T.assign(self.W_conv_3),self.b_conv_3_T.assign(self.b_conv_3),self.W_fc_1_T.assign(self.W_fc_1),self.b_fc_1_T.assign(self.b_fc_1),self.W_fc_2_T.assign(self.W_fc_2),self.b_fc_2_T.assign(self.b_fc_2)]
    
        self.create_training_method()
    
    def build(self):
        state_input = tf.placeholder(tf.float32, [None, 84, 84, 4])
        
        # Convolution layer 1
        W_conv_1 = weight_variable([8,8,4,32])
        b_conv_1 = bias_variable([32])
        conv_1_strides = [1,4,4,1]
        conv_1 = tf.nn.relu(tf.nn.conv2d(state_input, W_conv_1, strides=conv_1_strides, padding="VALID") + b_conv_1)     
        
        # Convolution layer 2
        W_conv_2 = weight_variable([4,4,32,64])
        b_conv_2 = bias_variable([64])
        conv_2_strides = [1,2,2,1]
        conv_2 = tf.nn.relu(tf.nn.conv2d(conv_1, W_conv_2, strides=conv_2_strides, padding="VALID") + b_conv_2)
        
        # Convolution layer 3
        W_conv_3 = weight_variable([3,3,64,64])
        b_conv_3 = bias_variable([64])
        conv_3_strides = [1,1,1,1]
        conv_3 = tf.nn.relu(tf.nn.conv2d(conv_2, W_conv_3, strides=conv_3_strides, padding="VALID") + b_conv_3)
        
        # Fully connected with ReLU
        W_fc_1 = weight_variable([7*7*64, 512])
        b_fc_1 = bias_variable([512])
        conv_3_flat = tf.reshape(conv_3, [-1, 7*7*64])
        fc_1 = tf.nn.relu(tf.matmul(conv_3_flat, W_fc_1) + b_fc_1)
        
        # Fully connected
        W_fc_2 = weight_variable([512,num_actions])
        b_fc_2 = bias_variable([num_actions])
        fc_2 = tf.matmul(fc_1, W_fc_2) + b_fc_2
        
        Q = fc_2
        
        return state_input,Q,W_conv_1,b_conv_1,W_conv_2,b_conv_2,W_conv_3,b_conv_3,W_fc_1,b_fc_1,W_fc_2,b_fc_2
        
    def create_training_method(self):
        self.action = tf.placeholder(tf.float32, [None, num_actions]) # One-hot vector
        self.Q_current = tf.reduce_sum(tf.mul(self.Q, self.action), axis=1) # NOTE: elementwise multiplication
        self.Q_target = tf.placeholder(tf.float32, [None])
        
        # Loss
        self.loss = tf.reduce_mean(tf.square(self.Q_target - self.Q_current));
        self.train_step = tf.train.RMSPropOptimizer(learning_rate=lr, decay=decay, momentum=momentum, epsilon=1e-6).minimize(self.loss)
                
    def copy_Q_net(self, sess):
        print "COPYING Q_NET"
        sess.run(self.copy_Q_net_op)
        
#    def step(self, sess, Q_target, seq, a):
#        """ Adjusts weights """
#        
#        sess.run(self.train_step, feed_dict={self.Q_target: Q_target, self.state_input: seq, self.action: a})
#        
#        if self.time_step % UPDATE_TIME == 0:
#            self.copy_Q_net()
            
    def train(self, sess, seq, seq_T, a, r):
        Q_vals_T = sess.run(self.Q_T, feed_dict={self.state_input_T: seq_T})
        Q_vals_T_max = np.max(Q_vals_T)
        Q_target = r + gamma*Q_vals_T_max
        
        # Temporary Q_target to simulate a batch of 1
        Q_target_temp = np.zeros(1)
        Q_target_temp[0] = Q_target
            
        sess.run(self.train_step, feed_dict={self.Q_target: Q_target_temp, self.state_input: seq, self.action: a})
        
        if self.time_step % UPDATE_TIME == 0:
            self.copy_Q_net(sess)
            
        return Q_target # For debug purposes
        
    #def add_to_replay(self, seq_current, action, reward, terminal, seq_next):
    def add_to_replay(self):
        """ Temporary since I need to increase time_step somehow """
        self.time_step += 1
        
    def get_action(self, sess, seq):
        """ Returns greedy action, seq is a preprocessed sequence """
        
        Q_vals = sess.run(self.Q, feed_dict={self.state_input: seq})
        return np.argmax(Q_vals)
        
    def get_Q_vals(self, sess, seq):
        """ Returns a vector with all Q-values for a given preprocessed sequence """
        
        Q_vals = sess.run(self.Q, feed_dict={self.state_input: seq})
        return Q_vals
        
    def get_Q_val(self, sess, seq, a):
        """ Returns the Q-value for a given preprocessed sequence and action """
        
        return sess.run(self.Q_current, feed_dict={self.state_input: seq, self.action: a})        

# Hyperparameters
# ACTION SPACE = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
num_actions = 4
num_episodes = 10000
gamma = 0.99
lr = 0.00025
decay = 0.99
momentum = 0.0
visualize_episodes = 10

INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.1
EPSILON_DECAY_TIME = 100000
EPSILON = INITIAL_EPSILON

UPDATE_TIME = 5000

# Run algorithm
env = gym.make("Breakout-v0")

with tf.Session() as sess:
    seq = Sequence(4)
    q_net = Q_Net()    
    
    init = tf.global_variables_initializer()
    sess.run(init)
    
    for i in range(num_episodes):
        print "EPISODE " + str(i) + " EPSILON: " + str(EPSILON)
        s = env.reset()
        phi = seq.add(s)
        
        while 1:
            if i%visualize_episodes == 0:
                env.render()
            
            # Temporary phi to simulate a batch of 1
            phi_temp = np.zeros([1,84,84,4])
            phi_temp[0,:,:,:] = phi
            
            # eps-greedy action selection
            if(np.random.rand() < EPSILON):
                a = np.random.randint(num_actions)
            else:
                a = q_net.get_action(sess, phi_temp)
                
            if EPSILON > FINAL_EPSILON:
                EPSILON -= (INITIAL_EPSILON - FINAL_EPSILON) / EPSILON_DECAY_TIME
            
            #Q_current = q_net.get_Q_val(sess, phi_temp, a)            
            
            s,r,d,_ = env.step(a)
            phi_new = seq.add(s)
            q_net.add_to_replay()
            
            # Temporary phi to simulate a batch of 1
            phi_new_temp = np.zeros([1,84,84,4])
            phi_new_temp[0,:,:,:] = phi_new
            
            # Calc max Q for next step                  
#            Q_vals = q_net.get_Q_vals(sess, phi_new_temp)
#            Q_vals_max = np.max(Q_vals)
#            Q_target = r + gamma*Q_vals_max   
#            
#            if i%visualize_episodes == 0:
#                print "taking action " + str(a)
#                print "Q_target = " + str(Q_target)
#            
#            # Temporary Q_target to simulate a batch of 1
#            Q_target_temp = np.zeros(1)
#            Q_target_temp[0] = Q_target
            
            # Make a one-hot
            a_onehot = np.zeros([1, num_actions])
            a_onehot[0,a] = 1
            
            #train(self, seq, seq_T, a, r):
            Q_target = q_net.train(sess, phi_temp, phi_new_temp, a_onehot, r)
            if i%visualize_episodes == 0:
                print "Q_target=" + str(Q_target)
            
            phi = phi_new
            
            # Step
            #q_net.step(sess, Q_target_temp, phi_temp, a_onehot)
        
#        plt.figure(1)
#        plt.gray()
#        plt.imshow(phi[:,:,0])
#        plt.figure(2)
#        plt.gray()
#        plt.imshow(phi[:,:,1])
#        plt.figure(3)
#        plt.gray()
#        plt.imshow(phi[:,:,2])
            
            #raw_input()
            
            if d:
                break
        
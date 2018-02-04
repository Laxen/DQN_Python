# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 12:28:04 2016

@author: alex
"""

import numpy as np
import scipy.misc as sp
import gym
import matplotlib.pyplot as plt

class State:
    def __init__(self, num_frames):
        self.num_frames = num_frames
        self.state = np.zeros([84,84,num_frames])
        
        # TODO: This could probably be done with deque, which might be faster
        
    def add(self, frame):
        frame_pp = self.preprocess(frame)
        
#        for i in reversed(range(self.num_frames-1)):
#            self.state[:,:,i+1] = self.state[:,:,i]
#        self.state[:,:,0] = frame_pp
        
        self.state = np.append(frame_pp, self.state[:,:,:-1], axis=2)
        
        return self.state
        
    def preprocess(self, frame):
        """ Preprocesses a frame and returns the result """
        
        # Extract luminance
        #lum = np.sum(frame, axis=2)
        #lum = np.divide(lum, 3)
        lum = frame[:,:,0]
        
        # Returns preprocessed frame
        pp = sp.imresize(lum, [110,84])  # Removes score and lives
        pp = (pp > 0)*255
        pp = np.expand_dims(pp, axis=2)
        return pp[26:110,:,:]

NUM_FRAMES = 4
action = 1

env = gym.make("Breakout-v0")

state = State(NUM_FRAMES)
env.reset()

for i in range(NUM_FRAMES):
    s,r,d,_ = env.step(action)
    state.add(s)
s,r,d,_ = env.step(action)
seq = state.add(s)

plt.gray()
for i in range(NUM_FRAMES):
    plt.subplot(1,NUM_FRAMES,i+1)
    plt.imshow(seq[:,:,i])
    
    
    
    
    
    
    
    
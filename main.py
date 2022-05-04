import matplotlib
import pybullet
import gym
import numpy as np
import matplotlib.pyplot as plt
import pybullet_data
import math
import rrtstar
import mpc
import time
import torch
import tf_util as U
from environment import Environment
from maddpg import MADDPGAgentTrainer
#import tensorflow as tf


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#import tensorflow.keras.layers as layers
from tf_slim.layers import layers

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    #model = tf.keras.models.Sequential()
    #model.add(tf.keras.Input(input))
    #model.add(tf.keras.layers.Dense(num_units, activation='relu'))
    #model.add(tf.keras.layers.Dense(num_units, activation='relu'))
    #model.add(tf.keras.layers.Dense(num_outputs))
    #return model
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out



model = mlp_model


arglist = {'scenario_name': 'two_line', 'max_episode_len': 10, 'num_episodes': 60000, 'num_adversaries': 0, 'good_policy': 'maddpg', 'lr': 1e-2, 'gamma': 0.95, 'batch_size': 1024, 'num_units': 64, 'collision_penalty': 20, 'action_duration' : 5}


with U.single_threaded_session():  
    env = Environment(arglist['collision_penalty'], arglist['action_duration'])
    agents = env.agents
    trainer = MADDPGAgentTrainer
    trainers = []
    obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
    for i in range(env.n):
        trainers.append(trainer(agents[i].name, model, obs_shape_n, env.action_space, i, arglist, 'False'))

    U.initialize()

    # initialize all rewards and actions and their respective lists
    episode_rewards = [0.0]
    agent_rewards = [[0.0] for _ in range(env.n)]
    final_ep_rewards = []
    final_ep_ag_rewards = []
    agent_info = [[]]
    saver = tf.train.Saver()
    obs_n = env.reset()
    episode_step = 0
    train_step = 0
    t_start = time.time()
    t_curr = time.time()
    current_episode = 0
    total_actions = []
    best_actions = []
    f = open('printOutput.txt', 'a')
    converges = 0

    while True:
        action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
        total_actions.append(action_n)
        new_obs_n, rew_n, done_n = env.step(action_n)
        episode_step += 1
        done = all(done_n)
        terminal = (episode_step >= arglist['max_episode_len'])
        for i, agent in enumerate(trainers):
            agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
        obs_n = new_obs_n

        for i, rew in enumerate(rew_n):
            episode_rewards[-1] += rew
            agent_rewards[i][-1] += rew

        # when task is completed, 
        if done or terminal:
            if done:
                converges += 1
                print("Completed task")
            else:
                converges = 0
            obs_n = env.reset()
            episode_step = 0
            print(episode_rewards[-1])
            best_actions = total_actions
            total_actions = []
            for a in agent_rewards:
                a.append(0)
            agent_info.append([[]])
            current_episode += 1
            #if current_episode % 100 == 0:
            #    t_curr = time.time() - t_curr
            #    print("current episode:", current_episode, "time:", t_curr, "rewards", episode_rewards[len(episode_rewards) - 2],file=f)
            #    plt.plot(episode_rewards)
            #    plt.title('total rewards')
            #    plt.ylabel('rewards')
            #    plt.xlabel('episodes')
            #    plt.savefig('rewardPlot' + str(current_episode) + '.png')
            #    plt.cla()

            episode_rewards.append(0)

        # plot rewards
        if converges == 10:
            plt.plot(episode_rewards)
            plt.title(str(arglist['scenario_name']) + ' rewards')
            plt.ylabel('rewards')
            plt.xlabel('episodes')
            plt.savefig('rewardPlot' + str(arglist['scenario_name']) + '.png')
            plt.cla()
            break

        train_step += 1

        loss = None
        for agent in trainers:
            agent.preupdate()
        for agent in trainers:
            loss = agent.update(trainers, train_step)
        
        # plot at the end of episodes
        if len(episode_rewards) > arglist['num_episodes']:
            plt.plot(episode_rewards)
            plt.title('total rewards')
            plt.ylabel('rewards')
            plt.xlabel('episodes')
            plt.savefig('finalRewardPlot.png')
            print('closed')
            f.close()
            break


    print(best_actions)


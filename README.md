# Continuous Control

## Table of Contents

1. [Introduction](#introduction)
2. [Directory Structure](#directoryStructure)
3. [Installation](#installation)
4. [Instructions](#instructions)
5. [Results](#results)

## Introduction <a name="introduction"></a>
<p align="center">
    <img src="./images/random_agent.gif" width="800" title="Random Agent" alt="Random Agent.">
</p>
In this environment, a double-jointed arm can move to target locations. 
A reward of +0.1 is provided for each step that the agent's hand is in the goal location. 
Thus, the goal of your agent is to maintain its position at the target location for as many time steps 
as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, 
and angular velocities of the arm. Each action is a vector with four numbers, corresponding to 
torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

## Directory Structure <a name="directoryStructure"></a>

- Root /
    - README.md (This readme file)
    - Report.md (A report describing results)
    - Continuous_Control.ipynb (The Jupyter notebook)
    - model.py (The neural network)
    - agent.py (The agent used for learning)
    - learning.py (A collection of methods for learning)
    - checkpoint_actor.pth (The neural network weights for the actor)
    - checkpoint_critic.pth (The neural network weights for the critic)
    - images /  
        - random_agent.gif  (Animated image of environment)
        - scores_plot.png (Plot of the scores during the learning process)
        
## Installation <a name="installation"></a>

This project was written in Python 3.6, using a Jupyter Notebook on Anaconda. Currently (Septemebr 2019) you cannot use Python 3.7, since tensorflow 1.7.1 doesn't have a version corresponding to python 3.7 yet.

The relevant Python packages for this project are as follows:

## Instructions <a name="instructions"></a>

Follow the instructions in Continuous_Control.ipynb to get started.

In order to train a network you have to create an agent:

agent = Agent(state_size=state_size, action_size=action_size, random_seed=random_seed)

Execute the learning method ddpg. (This will take around 80 minute when using GPU.)

scores = ddpg()

## Results <a name="results"></a>

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

Using the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) 
environment this projects trains a Actor Critic Policy Gradient Network agent to keep its hand near a goal location.
In this environment, a double-jointed arm can move to target locations. 
A reward of +0.1 is provided for each step that the agent's hand is in the goal location. 
Thus, the goal of your agent is to maintain its position at the target location for as many time steps 
as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, 
and angular velocities of the arm. 

Each action is a vector with four numbers, corresponding to 
torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

There are two separate versions of the environment:

- The first version contains a single agent.  
- The second version contains 20 identical agents, each with its own copy of the environment. 

This project solves the second version with 20 identical agents. 

In order to solve the environment the agents must get an average score of +30 (over 100 consecutive episodes 
and over all agents). 

After each episode the average of the scores for all 20 agents is computed. The environment is considered solved, 
when this average over 100 periods exceeds the threshold of 30. 

## Directory Structure <a name="directoryStructure"></a>

- Root /
    - README.md (This readme file)
    - Report.md (A report describing results)
    - Continuous_Control.ipynb (The Jupyter notebook)
    - model.py (The neural network)
    - agent.py (The agent used for learning)
    - checkpoint_actor.pth (The neural network weights for the actor)
    - checkpoint_critic.pth (The neural network weights for the critic)
    - images /  
        - random_agent.gif  (Animated image of environment)
        - scores_plot.png (Plot of the scores during the learning process)
        
## Installation <a name="installation"></a>

This project was written in Python 3.6, using a Jupyter Notebook on Anaconda. Currently (Septemebr 2019) you cannot use Python 3.7, since tensorflow 1.7.1 doesn't have a version corresponding to python 3.7 yet.

The relevant Python packages for this project are as follows:
 - numpy
 - torch
 - torch.nn
 - torch.nn.functional
 - torch.optim 
 - matplotlib.pyplot
 - unityagents
 
In order to use the code you have to download the environment from one of the links below.  

You need only select the environment that matches your operating system:

Version 2: Twenty (20) Agents
 - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip) 
 - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
 - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
 - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

Place the file in the GitHub repository and unzip (or decompress) the file. 
       
## Instructions <a name="instructions"></a>

Follow the instructions in Continuous_Control.ipynb to get started.

In order to train a network you have to create an agent:

agent = Agent(state_size=state_size, action_size=action_size, n_agents=num_agents, seed=random_seed)

Execute the learning method ddpg. (This will take around 80 minute when using GPU.)

scores = ddpg()

## Results <a name="results"></a>

The environment was solved in 100 episodes. An averaged score of 33.42 was reached. 
The score was averaged about the agents and the last 100 episodes. Below is a plot with the scores:

![scores](images/scores_plot.png)

A more detailed description of the results can be found in the Report.md file.


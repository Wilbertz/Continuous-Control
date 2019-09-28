# Project 2: Continuous Control

Author: [Harald Wilbertz](http://github.com/wilbertz) 

The report contains three parts:

- **Design and Implementation**
- **Results**
- **Future Improvements** 

## Design and Implementation

The basic algorithm lying under the hood is an actor-critic method. 
Policy-based methods like REINFORCE, which use a Monte-Carlo estimate, have the problem of 
high variance. TD estimates used in value-based methods have low bias and low variance. 
Actor-critic methods marry these two ideas where the actor is a neural network which
updates the policy and the critic is another neural network which evaluates the policy 
being learned which is, in turn, used to train the actor.


### Hyperparameters

  The code uses a lot of hyperparameters. The values a are given below

  | Hyperparameter                      | Value  |
  | ----------------------------------- | ------ |
  | Gamma (discount factor)             | 0.99   |
  
## Results

## Ideas for improvement


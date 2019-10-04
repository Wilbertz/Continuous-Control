# Project 2: Continuous Control

Author: [Harald Wilbertz](http://github.com/wilbertz) 

The report contains three parts:

- **Design and Implementation**
- **Results**
- **Future Improvements** 

## Design and Implementation
The design and implementation is heavily influenced by the paper [Continuous control with deep
reinforcement learning](https://arxiv.org/abs/1509.02971). In particular the networks architectures 
for both actor and critic and specific values for most hyperparameters are based on this paper.

The paper describes an algorithms that uses a Deep Deterministic Policy Gradient (DDPG) agent. 

This is a  model-free, off-policy actor-critic algorithm using deep function approximators 
that can learn policies in high-dimensional continuous action spaces. Model-free policy 
based learning algorithms are algorithms in which the agent learns
directly from the un-processed observation spaces without domain knowledge.

The Learning algorithm uses the Actor-Critic model in which the Critic model learns the value
function like DQN and uses it to determine how the Actor’s policy should change. 

The Actor brings the advantage of learning in continuous actions space
without the need for extra layer of optimization procedures required in a value based
function while the Critic supplies the Actor with knowledge of the performance.

This is different compared with DQN that learn indirectly through Q-values tables. The implemented 
algorithms learns from the observation spaces using policy gradients. 

The algorithms executes the following sequence of steps:
- A Replay buffer is created.
- Actor and Critic network are created and initialized. This includes Xavier initialisation of the hidden layers.
 The actor network does the policy approximation while the critic does the Q value estimation.
- The agent is trained for some episodes up to a maximum number of time-steps (max_t) in each episode, 
unless it terminates earlier.
- The training loop contains 2 processes: acting and learning.
- In the acting step, the agent uses the Actor network to get an action for a given state.
- In the learning step, the critic network is used as a feedback to the Actor network.
- Finally, actor and critic weights are updated from the current weights of the local actor and critic networks.

The algorithm uses two deep neural networks (actor-critic).
The Actor model is a neural network with 2 hidden layers with size of 400 and 300,
Tanh is used in the final layer that maps states to actions. Batch normalization is used for mini batch training.
The Critic model is similar to Actor model except the final layer is a fully connected layer that maps states and 
actions to Q-values.

The final layer weights and biases of both the actor and critic
were initialized from a uniform distribution [−3 × 10−3, 3 × 10−3] and [3 × 10−3, 3 × 10−3]. 
This was to ensure the initial outputs for the policy and value estimates were near zero. 

### Hyperparameters

  The code uses a lot of hyperparameters. The values a are given below

  | Hyperparameter                      | Value   |
  | ----------------------------------- | ------- |
  | Gamma (discount factor)             | 0.99    |
  | Tau                                 | 1e-3    |
  | Learning rate actor                 | 1e-3    |
  | Learning rate critic                | 1e-3    |
  | Batch size                          | 1024    |
  | Replay buffer size                  | 1000000 |

  
## Results
The environment was solved in 100 episodes. An averaged score of 31.06 was reached. 
The score was averaged over the agents and the last 100 episodes. Below is a plot with the scores:

![scores](images/scores_plot.png)

## Ideas for improvement

- Priority Experienced Replay  
The algorithm could be improved by using Priority Experienced Replay where important experience will be sampled more often:
[A novel DDPG method with prioritized experience replay](https://www.semanticscholar.org/paper/A-novel-DDPG-method-with-prioritized-experience-Hou-Liu/027d002d205e49989d734603ff0c2f7cbfa6b6dd).
This technique can reduce the training time and improve the stability of the learning process.

- Hyperparameter  
Last not least, hyperparameter search should improve the performance too.


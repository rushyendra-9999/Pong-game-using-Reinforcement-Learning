# Pong-game-using-Reinforcement-Learning
This repository contains a Python implementation of the classic Pong game enhanced with Q-learning agents. The project integrates the Pygame library for game development and Matplotlib for visualizing the learning progress.

# Problem Statement: 
The task is to create a Pong game environment where the agents learn to optimize their paddle movements to maximize ball hits within a limited number of episodes using Reinforcement learning techniques.
Reinforcement Learning technique used is Q-Learning.

State Space : State space refers to the various configurations that characterize the game state at any given moment. It can be defined based on the following components:
		1) Paddle Positions
		2) Ball velocity
		3) Ball position
		4) Score

Action Space : Action space refers to the set of actions available to the agents at any given state in the environment. 
	        It consists of the movements that each agent can perform to control their respective paddles.
	        Here Agent 1 and Agent 2 has same agent space (1) Move Up , 2) Move down 3) Stay Still)

Libraries required:
1) numpy
2) pygame
3) matplotlib
4) random

# Running the code:

1) Install a Python IDE such as Visual Studio Code or Pycharm. 
2) Run the code pong_main.py

# Output:
Game will run for nearly 20,000 episodes. After executing for 20,000 episodes a graph will appear on the screen.
The graph is between overall average scores of both the agents Vs Episodes. (Average is taken for every 10 episodes).

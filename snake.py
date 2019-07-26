import pygame, random, sys
import numpy as np
from pygame.locals import *
import pandas as pd 
import itertools 
import matplotlib 
import matplotlib.style
import matplotlib.pyplot as plt
from collections import defaultdict 
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
import gym
import gym_gridworlds
from gym import error, spaces, utils
from gym.utils import seeding

import dill as pickle
import codecs, json 
import time

matplotlib.style.use('ggplot') 


class SnakeEnv(gym.Env):
	metadata = {'render.modes': ['human']}
	
	def getState(self, xs,ys,applepos,sight, boardX, boardY, sWidth , dirs):
		#initialize state array
		matrixSize = 2*sight + 1;
		stateArray = np.zeros([matrixSize*matrixSize])
				
		def getXY(xview, yview, sight, sWidth, boardX, boardY, xs, ys, applepos):
			#check if out of board
			relX = (xview - sight)*(sWidth*2)
			relY = (yview - sight)*(sWidth*2)
			absX = xs[0] + relX
			absY = ys[0] + relY
			if(absX <= 0 or absX >= boardX):
				return -2
			elif(absY <= 0 or absX >= boardY):
				return -2
		
		#check if part of snake
			for snakeI in range(0, len(xs)):
				if(xs[snakeI] == absX and ys[snakeI] == absY):
					return -1
		
			#check if appleimage
			if(abs(applepos[0] - absX) < (sWidth*2) and abs(applepos[1] - absY) < (sWidth*2)):
				return 1
			return 0

		appleFound = False;
		for xview in range(0, matrixSize):
			for yview in range(0, matrixSize):
				val = getXY(xview, yview, sight, sWidth, boardX, boardY, xs, ys, applepos)
				stateArray[xview*matrixSize + yview] = val
				if(val > 1):
					appleFound = True;

		#Draw apple somewhere on the edge to help snake find it
		if(appleFound == False):
			#Find the edge where the apple is closest
			minDist = (self.boardX + self.boardY)**2 #start with max
			minIndX = -1
			minIndY = -1
			for xview in range(0, matrixSize):
				for yview in range(0, matrixSize):
					if( (xview > 0 and xview < matrixSize) and (yview > 0 and yview < matrixSize)):
						pass #not on the edge
					else:
						dist = (xview - self.applepos[0])**2 + (yview - self.applepos[1])**2
						if(minDist < dist):
							minDist = dist;
							minIndX = xview
							minIndY = yview
							
			stateArray[minIndX*matrixSize + minIndY] = 1

		stateTuple = tuple(stateArray.astype(int))
		return stateTuple
	
	def collide(self, x1, x2, y1, y2, w1, w2, h1, h2):
		if x1+w1>x2 and x1<x2+w2 and y1+h1>y2 and y1<y2+h2:return True
		else:return False
	
	def die(self, screen, score):
		if(self.renderOn):
			f=pygame.font.SysFont('Arial', 30);t=self.f.render('Your score was: '+str(self.score), True, (0, 0, 0));screen.blit(t, (10, 270));pygame.display.update();pygame.time.wait(1000);
		#print("You died. Final score: " + str(self.score) + " total ticks: " + str(self.ticks))
	
	def __init__(self):
		self.renderOn = True;
		self.leftRight = False;
		if(self.leftRight):
			self.action_index = 2
			self.action_space = spaces.Discrete(2);
		else:
			self.action_index = 3
			self.action_space = spaces.Discrete(4);
		self.eyesight = 2;
		self.sWidth = 10
		self.dirs = 0
		self.boardX = 200; #pixels
		self.boardY = 200; #pixels
		self.xs = [90, 90, 90, 90, 90];self.ys = [90, 70, 50, 30, 10];self.dirs = 0;self.score = 0;self.applepos = (random.randint(0, self.boardX - self.sWidth), random.randint(0, self.boardY - self.sWidth));pygame.init()
		self.s=pygame.display.set_mode((self.boardX, self.boardY));pygame.display.set_caption('Snake');self.f = pygame.font.SysFont('Arial', 20);self.clock = pygame.time.Clock()
		self.ticks = 0
		self.timeSinceLastFood = 0
		
	
	def step(self, action):
	#Returns: next_state, reward, done,
		if(self.renderOn):
			self.clock.tick(40)
		else:
			self.clock.tick()
		self.ticks = self.ticks + 1
		self.timeSinceLastFood = self.timeSinceLastFood + 1
		
		reward = 0
		#if first apple is eaten - try to give negative rewards to speed up finding
		if(self.timeSinceLastFood > 50):
			reward = -0.1
		
		if(self.leftRight):
			if(action > 0):
				if(action == 1): #turn right
					self.dirs = self.dirs + 1;
					if(self.dirs >= 4):
						self.dirs = 0
				elif(action == 2): #turn left
					self.dirs = self.dirs - 1;
					if(self.dirs < 0):
						self.dirs = 3
		else:
			if (action == 0) and self.dirs != 0:self.dirs = 2 #UP
			elif (action == 1) and self.dirs != 2:self.dirs = 0 #DOWN
			elif (action == 2) and self.dirs != 1:self.dirs = 3 #LEFT
			elif (action == 3) and self.dirs != 3:self.dirs = 1 #RIGHT
		
		#Manual play
		for e in pygame.event.get():
			if e.type == QUIT:
					sys.exit(0)
			if e.type == KEYDOWN:
				if e.key == K_UP and dirs != 0:dirs = 2
				elif e.key == K_DOWN and dirs != 2:dirs = 0
				elif e.key == K_LEFT and dirs != 1:dirs = 3
				elif e.key == K_RIGHT and dirs != 3:dirs = 1
				elif e.key == K_q:sys.exit(0)
		i = len(self.xs)-1
		while i >= 2:
			if self.collide(self.xs[0], self.xs[i], self.ys[0], self.ys[i], 20, 20, 20, 20):
				self.die(self.s, self.score)
				reward = -100;
				next_state = self.getState(self.xs,self.ys,self.applepos,self.eyesight, self.boardX, self.boardY, self.sWidth, self.dirs)
				return next_state, reward, True, None
			i-= 1
		if self.collide(self.xs[0], self.applepos[0], self.ys[0], self.applepos[1], 20, 10, 20, 10):
			reward = +100;
			self.timeSinceLastFood = 0
			self.score+=1;self.xs.append(700);self.ys.append(700);
			#make sure we don't place the apple over the snake
			newApplePosFound = False
			attempts = 0;
			while(newApplePosFound == False or attempts > 200):
				attempts = attempts + 1
				self.applepos=(random.randint(0,self.boardX - 10),random.randint(0,self.boardY - 10))
				for applei in range(1,len(self.xs)):
					if not self.collide(self.xs[applei], self.applepos[0], self.ys[applei], self.applepos[1], 20, 10, 20, 10):
						newApplePosFound = True;
						break;
		
		#starve to death
		if self.timeSinceLastFood > 1000:
			self.die(self.s, self.score)
			reward = -100;
			next_state = self.getState(self.xs,self.ys,self.applepos,self.eyesight, self.boardX, self.boardY, self.sWidth, self.dirs)
			return next_state, reward, True, None
			
		if self.xs[0] < 0 or self.xs[0] > (self.boardX - 2* self.sWidth) or self.ys[0] < 0 or self.ys[0] > (self.boardY - 2* self.sWidth):
			self.die(self.s, self.score)
			reward = -100;
			next_state = self.getState(self.xs,self.ys,self.applepos,self.eyesight, self.boardX, self.boardY, self.sWidth, self.dirs)
			return next_state, reward, True, None
		i = len(self.xs)-1
		while i >= 1:
			self.xs[i] = self.xs[i-1];self.ys[i] = self.ys[i-1];i -= 1
		if self.dirs==0:self.ys[0] += 20
		elif self.dirs==1:self.xs[0] += 20
		elif self.dirs==2:self.ys[0] -= 20
		elif self.dirs==3:self.xs[0] -= 20	
		
		next_state = self.getState(self.xs,self.ys,self.applepos,self.eyesight, self.boardX, self.boardY, self.sWidth, self.dirs)
		# Game continues, crack on
		return next_state, reward, False, None
	
	def reset(self):
		self.xs = [90, 90, 90, 90, 90];self.ys = [90, 70, 50, 30, 10];self.dirs = 0;self.score = 0;self.applepos = (random.randint(0, (self.boardX - self.sWidth)), random.randint(0, (self.boardY - self.sWidth)));
		self.clock = pygame.time.Clock()
		self.ticks = 0
		self.timeSinceLastFood = 0
		if(self.renderOn):
			pygame.init();
			self.s=pygame.display.set_mode((self.boardX, self.boardY));
			pygame.display.set_caption('Snake');
		
		state = self.getState(self.xs,self.ys,self.applepos,self.eyesight, self.boardX, self.boardY, self.sWidth, self.dirs)
		return state
	
	def render(self, mode='human'):
		if(self.renderOn):
			self.f = pygame.font.SysFont('Arial', 20)
			appleimage = pygame.Surface((10, 10));appleimage.fill((0, 255, 0));img = pygame.Surface((20, 20));img.fill((255, 0, 0))
			self.s.fill((255, 255, 255))	
			for i in range(0, len(self.xs)):
				self.s.blit(img, (self.xs[i], self.ys[i]))
			self.s.blit(appleimage, self.applepos);self.t=self.f.render(str(self.score), True, (0, 0, 0));self.s.blit(self.t, (self.sWidth, self.sWidth));pygame.display.update()
		
	def close(self):
		...
	
env = SnakeEnv();
	
def createEpsilonGreedyPolicy(Q, epsilon, num_actions): 
	def policyFunction(state): 
 
		Action_probabilities = np.ones(num_actions, dtype = float) * epsilon / num_actions 
							
		best_action = np.argmax(Q[state]) 
		Action_probabilities[best_action] += (1.0 - epsilon) 
		return Action_probabilities 
 
	return policyFunction 
	
def qLearning(env, num_episodes, start_episode = 0, discount_factor = 0.9, alpha = 0.7, epsilon = 0.01): 

	epsilon_min = 0.01;
	epsilon_decay = 0.95;
	
	fileName = "model.dat"
	Q = defaultdict(lambda: np.zeros(env.action_space.n))		
	
	try:
		with open(fileName, 'rb') as handle:
			P = pickle.load(handle)
			Q.update(P)
		print("Loaded stored model")
	except:
		print("Loading default model")
			
	# Create an epsilon greedy policy function 
	# appropriately for environment action space 
	policy = createEpsilonGreedyPolicy(Q, epsilon, env.action_space.n)
	episode_lengths = np.zeros(num_episodes)
	episode_rewards = np.zeros(num_episodes)
	
	# For every episode 
	for ith_episode in range(start_episode, start_episode+num_episodes):
		iIndex = ith_episode-start_episode
		#epsilon_current = max(epsilon_min, epsilon*(epsilon_decay**(ith_episode+1)))
		# Reset the environment and pick the first action 
		state = env.reset() 
		
		if((ith_episode+1) % 5000 == 0):
			print("Saving graph")	
			xrange = range(0, iIndex)
			N = 100
			rewardsAvg = pd.Series(episode_rewards).rolling(window=N).mean().iloc[N-1:].values
			plt.plot(xrange, episode_rewards[0:iIndex],'.b',range(0, len(rewardsAvg)), rewardsAvg,'--r')
			plt.ylabel('rewards')
			plt.xlabel('episode')
			plt.savefig("graph_" + str(ith_episode+1) + ".png")
			
		if((ith_episode+1) % 25000 == 0):
			print("Saving model")
			with open("model_" + str(ith_episode+1) + ".dat", 'wb') as handle:
				pickle.dump(dict(Q), handle, protocol=pickle.HIGHEST_PROTOCOL)		

		for t in itertools.count(): 
			# get probabilities of all actions from current state 
			action_probabilities = policy(state) 
	 
			# choose action according to	
			# the probability distribution 
			action = np.random.choice(np.arange(len(action_probabilities)),p = action_probabilities)
			
			try:
				env.render()
			except:
				pass
			next_state, reward, done, _ = env.step(action) 
			
			episode_rewards[iIndex] += reward
			episode_lengths[iIndex] = t
			
			# TD Update 
			best_next_action = np.argmax(Q[next_state])		 
			td_target = reward + discount_factor * Q[next_state][best_next_action] 
			td_delta = td_target - Q[state][action] 
			Q[state][action] += alpha * td_delta 
			# done is True if episode terminated		
			if done: 
				break
									 
			state = next_state 
		
		averagePrint = 500
		if((ith_episode+1) % averagePrint == 0):
			avgReward = np.mean(episode_rewards[iIndex+1-averagePrint:iIndex])
			print(str(ith_episode+1) + " of " + str( start_episode+num_episodes) + ": Reward=" + str(avgReward))
			
	print("Saving model")
	with open(fileName, 'wb') as handle:
		pickle.dump(dict(Q), handle, protocol=pickle.HIGHEST_PROTOCOL)		 
	return Q, episode_rewards, episode_lengths, 
		

start = time.time()
start_episode = 0
num_episodes = 10000
Q, episode_rewards, episode_lengths = qLearning(env, num_episodes, start_episode)
xrange = range(0, num_episodes)

plt.plot(xrange, episode_lengths,'--r', xrange, episode_rewards,'.b')
plt.ylabel('rewards')
plt.xlabel('episode')
plt.show()
np.savetxt('episode_lengths.out', episode_lengths, delimiter=',') 
np.savetxt('episode_rewards.out', episode_rewards, delimiter=',') 
end = time.time()
print("Final score: " + str(np.mean(episode_rewards[-1000:])))
print("Time elapsed: " + str(end - start))
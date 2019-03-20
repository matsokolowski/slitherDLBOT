import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D,MaxPooling1D,AveragePooling1D, Dropout, Flatten,Reshape, LeakyReLU , Subtract,Add,Average,add,subtract,average,Lambda,GaussianNoise,RepeatVector,Multiply,Concatenate,BatchNormalization,Activation
from keras.models import Sequential
from keras import backend as K, Model
from keras.callbacks import TensorBoard
from collections import deque
from keras.optimizers import Adam
from threading import Thread
from keras.constraints import max_norm

import numpy as np
import random
import time

from env import environment

si =  lambda x: x/(1+abs(x))

bnormed_relu = lambda x: Activation("relu")(BatchNormalization()(x))

class slitherBot:
	def __init__(self):

		self.memory = deque(maxlen=3072)

		self.input_shape=(96,96,1)
		#self.gamma = 0.8    # discount rate
		self.gamma = 0.75    # discount rat
		self.epsilon = 0.66  # exploration rate
		self.epsilon_min = 0.015
		self.epsilon_decay = 0.96
		#self.learning_rate = 0.00025
		self.learning_rate = 0.00001
		self.action_size = 9
		self.fitqueue = []
		
		self.dual = False

		self.m_i = 0
		self.m_l = 0
		self.m_T = 0

		self.fq = 0

		self.build_vision_model()


		self.Q1 = self.build_duelingQ_model() 
		if self.dual:
			self.Q2 = self.build_duelingQ_model()
			self.fitQ2 = lambda: self.Q2.set_weights(self.Q1.get_weights())
			self.fitQ2()


	def act(self, state):
		if np.random.rand() <= self.epsilon:
			if self.m_i > self.m_T: 
				self.m_l = random.randrange(self.action_size)
				self.m_i = 0
				if self.m_l == 8: r = 5
				else: r = 100
				self.m_T = random.sample(range(r),1)[0]
			else: self.m_i += 1
			return self.m_l

		act_values = self.Q1.predict(state)
		#print(act_values)
		return np.argmax(act_values[0])

	def build_vision_model(self):
		self.state_input = Input(shape=self.input_shape
)
		x = Conv2D(16, (2, 2))(self.state_input)
		x = bnormed_relu(x)

		x = Conv2D(32, (3, 3))(x)
		x = bnormed_relu(x)

		x = MaxPooling2D(pool_size=(3, 3))(x)

		x = Conv2D(48, (3, 3))(x)
		x = bnormed_relu(x)

		x = MaxPooling2D(pool_size=(3, 3))(x)

		x = Conv2D(16, (3, 3))(x)
		x = bnormed_relu(x)
		
		output = Flatten()(x)		
		self.vision_model = output

	def build_duelingQ_model(self):

		def define_multilayer_critic(x):
			#with souble critic; 128 - 1st; 32 - 2ed good performance
			x = Dense(256, activation='relu')(x)

			critic = [
				Dense(64, activation='relu'),
				Dense(1, activation='linear'),
			]

			def buildcritic(x):
				for r in range(len(critic)): x = critic[r](x)
				return x

			l = []

			s = K.int_shape(x)[1]
			for a in range(self.action_size):

				z = np.ones(self.action_size * 2)
				m = np.zeros(self.action_size)

				MOnes = np.ones(s - self.action_size)
				M = np.concatenate((MOnes, m)).reshape(1,s) ## <- mask
				m[a] = 1.0
				Mv = np.concatenate((MOnes*0, m)).reshape(1,s) ## <- value
				z[ a * 2 ] = 0.0
				z = np.tile(z,int(s/len(z))+1)[:s].reshape(1,s)

				b = Lambda(lambda x: K.constant(z) * x * K.constant(M) + K.constant(Mv) )(x)

				l.append( buildcritic(b) )
			return Concatenate()(l)

		I = self.vision_model

		d = Dense(384, activation='relu')(I)
		#d = Dropout(0.2)(d)
		V = define_multilayer_critic(d)

		def flat(d,V):
			d = Dense(256, activation='relu')(Lambda(K.concatenate)([d,V]))
			#d = Dense(128, activation='relu')( d )
			d = Dense(48, activation='relu',kernel_constraint = max_norm(0.005,axis=-1), bias_constraint = max_norm(3))( d ) 

			return Dense(self.action_size, activation='linear', kernel_constraint = max_norm(0.001,axis=-1))(d)

		A = flat(d,V)


		outp = lambda x: x[0] - K.mean(x[0]) + x[1] #+ x[2]
		out = Lambda(outp, output_shape = (self.action_size,))([A, V])
		m = Model(input=self.state_input ,output = out)
		m.summary()
		m.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))

		return m


		
	def fitq(self):
		try:
			q=self.fitqueue.pop()
			self.fit(*q)
		except: pass

	def fit(self,state, action, reward, next_state, done):
			Q1 = self.Q1
			if self.dual : t = self.Q2
			else : t = self.Q1

			target_f = Q1.predict(state) 			

			target = reward
			d = abs( target_f[0][action] - reward )

			p = np.amax(t.predict(next_state)[0])

			if not done:
				target = reward + ( self.gamma * p )
			target_f[0][action] = min(target,3)
			print("target:", target, p,"mean:", np.mean(target_f))
			Q1.fit(state, target_f, epochs=1, verbose=0, sample_weight = np.array([0.01,]) )

			
			if self.dual:
				if self.fq > 50 :
					self.fitQ2()
					self.fq = 0
				self.fq += 1

			return d


	def replay(self, batch_size, live=False):
		minibatch = random.sample(self.memory, min(len(self.memory),batch_size))
		print ("batch length", len(minibatch),len(self.memory))
		if live: fit = lambda *a: self.fitqueue.append(a)
		else: 
			fit = lambda *a,ap = self.mem_by_diffirence.append: ap( (a, self.fit(*a) ) )

		for b in minibatch: fit(*b)

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

	def prioratized_replay(self,size = 512, ntimes = 7):
		try: states, actions, rewards, next_states, dones = \
			[ np.array(x) for x in zip(*self.memory) ]
		except: return
		chunk = 512
		#next time try batch 50

		for k in range(ntimes):

			predictions = []

			## state preditions
			for x in range(0,len(states),chunk):
				s = np.squeeze(states[x : x + chunk],axis=1)
				predictions += self.Q1.predict(s).tolist()

			#differences between reward and prediction

			diff = np.take(predictions, np.arange(len(states)) * self.action_size + actions)
			diff = np.abs(diff - rewards)

			a = diff

			# Calcualte the next states
			predictions_next = []

			for x in range(0,len(a),chunk):
				s = np.squeeze(next_states[x : x + chunk],axis=1)
				predictions_next += self.Q1.predict(s).tolist()

			max_in_next = np.amax(predictions_next,axis=1)

			# Calculating the targets
			targets = (max_in_next * 0.95 + rewards)
			targets = np.where(dones,targets,rewards).reshape(len(a),1)
			places = np.tile(np.arange(self.action_size),(len(a),1)) - actions.reshape(len(a),1) == 0
			targets_f = np.where(places, targets, predictions,)
			targets_f[ targets_f > 5 ] = 5

			s = np.squeeze(states,axis=1)

			#self.Q1.fit(s, targets_f , batch_size = 100, epochs = 2,shuffle=True,sample_weight = (diff + 1))
			self.Q1.fit(s, targets_f , batch_size = 100, epochs = 2,shuffle=True)


		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

	def remember(self, *e):
		self.memory.append(e)

if __name__ == "__main__":

	e = environment()
	agent = slitherBot()
	e.start()


	reward = 0
	r = rr = 1
	c = 0
	state = e._frame()
	
	delay = 3
	states = deque(maxlen=delay)

	while not int(e.score()): time.sleep(1)

	t1 = t = time.time()
	dt= 0 

	while True:

		t1 = time.time()
		dt = t1 - t
		haste  = abs(dt - 0.066)
		if haste < 0 :  
			time.sleep( haste )
			t1 += haste
			
		action = agent.act(state)

		e.action(action)
		#time.sleep(0.06)
		state = e._frame()


		rr =  float(e.score())
		reward = si(rr - r)
		r = rr

		print("action", action , reward, dt,rr)
		
		t = t1


		states.append( [state, action, '0', '0', rr > 0, ] )

		if len(states) < delay : continue

		states[-2][2] = reward
		states[-2][3] = state

		if '0' not in states[0]:
			agent.remember( *states[0] )

		"""if c > 150:
			agent.replay(50,True)
			c = 0
		else:
			agent.fitq()
			c += 1
"""
		if rr == 0:
			#roll death back
			for x in range(10):
				agent.memory.pop()
			states.clear()
			agent.memory[-1] = agent.memory[-1][:2] + (rr,) + agent.memory[-1][3:]
			
			# train the netwoek
			agent.prioratized_replay()
			print ("starting", agent.epsilon)
			e.start()
			while not int(e.score()): time.sleep(1)
			rr = 1



import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D,MaxPooling1D,AveragePooling1D, Dropout,AveragePooling2D, Flatten,Reshape, LeakyReLU , Subtract,Add,Average,add,subtract,average,Lambda,GaussianNoise,RepeatVector,Multiply,Concatenate,BatchNormalization,Activation
from keras.models import Sequential
from keras import backend as K, Model
from keras.callbacks import TensorBoard
from collections import deque
from keras.optimizers import Adam
from threading import Thread
from keras.constraints import max_norm
from keras.models import load_model

import numpy as np
import random
import time
import os
import pickle
from env import environment

si =  lambda x: x/(1+abs(x))

bnormed_relu = lambda x: Activation("relu")(BatchNormalization()(x))

def crop(dimension, start, end):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
    return Lambda(func)

class slitherBot:
    def __init__(self):

        self.memory = deque(maxlen=1800)
        self.recordIntoFiles = True;
        self.recordAnchor = np.random.rand(100)

        self.input_shape=(64,64,1)
        #self.gamma = 0.8    # discount rate
        self.gamma = 0.8    # discount rat
        self.epsilon = 0.1  # exploration rate
        self.epsilon_min = 0.015
        self.epsilon_decay = 0.96
        #self.learning_rate = 0.00025
        self.learning_rate = 0.000001
        self.action_size = 9
        self.fitqueue = []
        self.trained = False
        self.dual = False

        self.m_i = 0
        self.m_l = 0
        self.m_T = 0

        self.fq = 0

        self.build_vision_model()
        self.Q1 = self.build_duelingQ_model()

        try:
            if os.path.isfile("weights/Q.weights"):
                self.Q1.set_weights( pickle.load(open("weights/Q.weights","rb")) )
                print("model successfuly loaded.")
            else:
                print("loading model failed.")
                self.save()
        except: pass

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
        self.state_input = Input(shape=self.input_shape,name="input")

        x = Conv2D(32, (4, 4), activation="relu")(self.state_input)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(32, (4, 4), activation="relu")(x)
        x = Conv2D(48, (4, 4), activation="relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(64, (3, 3), activation="relu")(x)
        x = AveragePooling2D(pool_size=(2, 2))(x)

        print ( K.int_shape(x) )
        output = x # Flatten()(x)        
        self.vision_model = output

    def save(self):
        #self.vision_model.save('vision.h5')
        pickle.dump(self.Q1.get_weights(),open("weights/Q.weights","wb"))

    def build_duelingQ_model(self):

        def define_multilayer_critic(i):

            critic = [
                Dense(384, activation='relu'),#Dense(256, use_bias=False), #- former 256
                Dense(256, activation='relu'), #- former 64
                #Dense(48, activation='relu'), # <-remove for old performance
                ( Dense(32, activation='relu'), Dense(32, activation='relu') ),# <-remove for old performance
                ( Dense(24, activation='relu'), Dense(24, activation='relu') ),
                ( Dense(1, activation='linear'), Dense(1, activation='linear') ),
            ]
            
            def buildcritic(x):
                x = critic[0](x)
                x = critic[1](x)
                #x = critic[2](x)
                
                #x, y =  critic[2][0](x) , critic[2][1](x) 
                x, y = critic[2][0](x), critic[2][1](x) 
                x, y = critic[3][0](x), critic[3][1](y)
                x, y = critic[4][0](x), critic[4][1](x) 


                #x = critic[4][0](x)
                #y = critic[4][1](y)


                return x,y

            l = []
            n = []

            s = K.int_shape(i)[1]
            for a in range(self.action_size):

                z = np.ones(self.action_size * 2)
                m = np.zeros(self.action_size)

                MOnes = np.ones(s - self.action_size)
                M = np.concatenate((MOnes, m)).reshape(1,s) ## <- mask
                m[a] = 5.0
                Mv = np.concatenate((MOnes*0, m)).reshape(1,s) ## <- value
                z[ a * 2 ] = 0.0
                z = np.tile(z,int(s/len(z))+1)[:s].reshape(1,s)

                b = Lambda(lambda x: K.constant(z) * x * K.constant(M) + K.constant(Mv) )(i)

                x = buildcritic(b)
                l.append(x[0])
                n.append(x[1])

            return Concatenate()(l), Concatenate()(n)

        I = self.vision_model

        """
        CapsuleLayer = Dense(64, activation='relu')
        cropedInputVerticly = []
        cropedInput = []

        for x in range(0,9,3):
            cropper = crop(1,x,x+3)
            c = cropper(I)
            cropedInputVerticly.append(c)

        for x in range(0,9,3):
            cropper = crop(2,x,x+3)
            for y in cropedInputVerticly:
                cropedInput.append( cropper(y) ) 
       
        capsules = list([ CapsuleLayer( Flatten()(x) ) for x in cropedInput ])
        capsules = np.array(capsules).reshape((3,3))

        capsuleLayer_2 = Dense(128, activation='relu')
 
        slices = [
            capsules[:2, :2],
            capsules[1:3, :2],
            capsules[:2, 1:3],
            capsules[1:3, 1:3],
        ] 
        outputs = []
        
        for x in slices:
            x = list( x.flatten() )
            x = Concatenate()( x )
            outputs.append(capsuleLayer_2(x))
        d = Concatenate()(outputs)
        """
        winHeight = 2
        croppersX = [ crop(1,x, x + winHeight) for x in range( K.int_shape(I)[1] - winHeight + 1) ]
        croppersY = [ crop(2,x, x + winHeight) for x in range( K.int_shape(I)[2] - winHeight + 1) ]
        Layer = Dense(96, activation='relu') 
        windows = [[],]
        for x in croppersX:
            for y in croppersY:
                d = Layer( Flatten()( y(x(I)) ) )
                windows[-1].append( d )
            windows.append([])

        winHeight = 2
        slicesX = [ windows[x: x + winHeight] for x in range( len(windows) - winHeight + 1) ]
        slicesXY = []
        for x in slicesX:
            for y in range( len(x[1]) - winHeight + 1 ):
                slicesXY.append(np.array(x)[:,y: y+2].flatten().tolist())
        
        Layer2 = Dense(96, activation='relu') 

        d = list([ Layer2( Concatenate()(x) ) for x in slicesXY ])
        d = Concatenate()( d )
        #print("concatenated",K.int_shape(d))
        V, A = define_multilayer_critic(d)


        def outp(x):
            u  = (x[0] - K.mean(x[0]))
            return u + x[1] 
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

    def replay_recorded(self):
        a = pickle.load(open('bestscores.pk',"rb"))[:150]
        np.random.shuffle(a)

        irc = self.recordIntoFiles
        self.recordIntoFiles = False

        l = len(a)

        for (f,s),i in zip(sorted(a,lambda x: x[0]),range(l)):
            print("%d/%d - %d" % (i,l,s),f)
            try: 
                f = open(f,"rb")
                pack = pickle.load(f)
            except: continue
            self.prioratized_replay( 512, 7s, pack )
            f.close()
        self.recordIntoFiles = irc

    def prioratized_replay(self,size = 512, ntimes = 10, pack = False):
        try:
            states, actions, rewards, next_states, dones = pack = \
                pack or [ np.array(x) for x in zip(*self.memory) ]
        except: return
        
        if self.recordIntoFiles :        
            i = states.tobytes().find(self.recordAnchor.tobytes())
            if i < 0:
                pickle.dump(pack,open( "recorded/%s.pk" % time.time(),"wb"))
                self.recordAnchor = states[-1]
                self.trained = False
            elif  i / len(states.tobytes()) < 0.4 and not self.trained:
                self.trained = True
            else: return

        chunk = 512
        #next time try batch 50

        for k in range(ntimes):

            predictions = []

            ## state preditions
            for x in range(0,len(states),chunk):
                s = np.squeeze(states[x : x + chunk],axis=1)
                predictions += self.Q1.predict(s).tolist()

            #differences between reward and prediction

            actions = actions.astype('int64')
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
            targets = (max_in_next * 0.97 + rewards)
            targets = np.where(dones,targets,rewards).reshape(len(a),1)
            places = np.tile(np.arange(self.action_size),(len(a),1)) - actions.reshape(len(a),1) == 0
            targets_f = np.where(places, targets, predictions,)
            targets_f[ targets_f > 5 ] = 5

            s = np.squeeze(states,axis=1)

            #self.Q1.fit(s, targets_f , batch_size = 200, epochs = 1,shuffle=True,sample_weight = (diff + 1))
            self.Q1.fit(s, targets_f , batch_size = 200, epochs = 1,shuffle=True)


        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def remember(self, *e):
        self.memory.append(e)

if __name__ == "__main__":
    e = environment()
    agent = slitherBot()

    ## starting webpage and game
    e.start()
    e.points = 0

    reward = 0
    r = rr = 1
    c = 0
    state = e._frame()
    
    delay = 5
    states = deque(maxlen=delay)
    def restart():
        e.initpage()
        e.start()

    while True:
        retries = 0
        while not int(e.points):
            e.action(1)
            time.sleep(1)
            if retries > 10:
                restart()
                retries = 0
            retries +=1

        t1 = t = time.time()
        dt = 0 

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
            states[-4][2] = reward
            states[-2][3] = state

            if '0' not in states[0]:
                if states[0][1] == 8:
                    states[0][2] -= 0.166
                    print("punishment")
                agent.remember( *states[0] )

            if rr == 0:
                #roll death back
                #try:
                #    for x in range(10):
                #        agent.memory.pop()
                #except: continue
                states.clear()
                agent.memory[-1] = agent.memory[-1][:2] + (rr,) + agent.memory[-1][3:]
                # train the netwoek
                if agent.dual: agent.fitQ2()
                if len(agent.memory) > 900:
                    agent.prioratized_replay()
                    agent.save()

                # starting the bot
                print ("starting", agent.epsilon)
                restart()
                e.points = 0
                rr = 1
                break



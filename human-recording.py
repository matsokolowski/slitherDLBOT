from collections import deque
import numpy as np
import random
import time
import os
import pickle
from env import environment

si =  lambda x: x/(1+abs(x))

if __name__ == "__main__":

    e = environment()
    e.driver.maximize_window()
    e.points = 0
    delay = 4

    reward = 0
    r = rr = 1
    state = e._frame()
    states = deque(maxlen=delay)

    t1 = t = time.time()
    dt = 0
    
    memory = []

    while True:
        t1 = time.time()
        dt = t1 - t
        haste  = abs(dt - 0.066)
        if haste < 0 :  
            time.sleep( haste )
            t1 += haste
        t = t1
 
        print("action ", end = '')

        action, score = e.get_score_and_last_player_move()

        print(action,score,dt)

        if not score:
            time.sleep(1) 
            continue

        state = e._frame()

        rr =  float(e.score())
        reward = si(rr - r)
        r = rr

        states.append( [state, action, '0', '0', rr > 0, ] )

        if len(states) < delay : continue
    
        states[-4][2] = reward
        states[-2][3] = state

        if '0' not in states[0]:
            if states[0][1] == 8:
                states[0][2] -= 0.166
                print("punishment")

            memory.append( states[0] )

        if len(memory) < 2048:
            continue

        pickle.dump( \
            [ np.array(x) for x in zip(*memory) ], \
            open( "recorded_human/%f.pk" % time.time(),\
            "wb" )
        )
        del memory[:]
            


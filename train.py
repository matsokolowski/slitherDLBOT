#!/usr/bin/python
from model import slitherBot
if __name__ == "__main__":
    agent = slitherBot()
    ##replaying recorded
    for f in range(2): 
        agent.replay_recorded()
        agent.save()

#!/usr/bin/python
import pickle
import os
 
def get():
    a = []
    for g in ( "./recorded_human/", "./recorded/" ):
        #for g in ["./recorded_human/" ,]:
        for dr, dirs, files in os.walk(g):
            for n in files:
                try:
                    f = open(dr + n,"rb")
                    pack = pickle.load(f)
                    f.close()
                except: continue
                print(dr + n)
                a.append( (dr + n , sum( abs(pack[2]) ) ) )

    a.sort(key = lambda x: x[1])
    a.reverse()
    return a

def deleteLowScores(limit, scores):
    whiteList = next(zip(*scores[:limit]))
    for g in ( "./recorded_human/", "./recorded/" ):
        for dr, dirs, files in os.walk(g):
            for n in files:
                dn = dr + n
                if dn not in whiteList:
                     os.remove(dn)
                else: print("inside",dn)


if __name__ == "__main__":
    l = get()
    pickle.dump(l,open("bestscores.pk","wb"))
    deleteLowScores(300,l)

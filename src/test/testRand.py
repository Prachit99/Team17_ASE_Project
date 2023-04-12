import sys
sys.path.insert(1, "../")
from Utils import rint


def testRand():
     Seed=1
     t=[]
     for i in range(1,10):
          t.append(rint(0,100,Seed))

     Seed=1
     u=[]
     for i in range(1,10):
          u.append(rint(0,100,Seed))

     print(u,t)
     for k,v in enumerate(t):
          assert(v==u[k])
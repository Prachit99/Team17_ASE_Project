import sys
sys.path.insert(1, "../")
import Num
import Constants
from Utils import rand,rnd


def testNum():
     num1,num2 = Num.Num(),Num.Num()
     for i in range(1,10001):
          num1.add(rand(0,1))
     for i in range(1,10001):
          num2.add(pow(rand(0,1),2))
     m1,m2=rnd(num1.mid(),1),rnd(num2.mid(),1)
     d1,d2 = rnd(num1.div(),1), rnd(num2.div(),1)
     print(1, m1, d1)
     print(2, m2, d2) 
     return m1 > m2 and 0.5 == rnd(m1)





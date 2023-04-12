from Constants import Constants
from Data import Data
from Num import Num
from Utils import  oo,rnd


def testDist():
    file = Constants().file
    data = Data(file)
    num = Num()
    for row in data.rows:
        num.add(Data.dist(data, row, data.rows[1]))
    print(num.lo, num.hi, rnd(num.mid()), rnd(num.div()))
    
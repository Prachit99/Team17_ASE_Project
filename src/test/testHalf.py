from Constants import Constants
from Data import Data
from Utils import o


def testHalf():
    file = Constants().file
    data = Data(file)
    left,right,A,B,mid,c = data.half()
    l,r= data.clone(left),data.clone(right)
    print("l",o(l.stats('mid', l.Cols.y, 2)))
    print("r",o(r.stats('mid', r.Cols.y, 2)))


    

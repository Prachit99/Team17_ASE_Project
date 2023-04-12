import Constants
import Data
from Utils import  oo


def testData():
    file = Constants.Constants().file
    data = Data.Data(file)
    col = data.cols.x[0]
    print(col.lo, col.hi, col.mid(), col.div())
    oo(data.stats('mid', data.cols.y, 2))



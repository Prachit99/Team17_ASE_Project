import Constants
import Data
from Utils import  oo


def testData():
    file = Constants.Constants().file
    data = Data.Data(file)
    col = data.Cols.x[0]
    print(type(col))
    # print(f'low: {col.lo}, high: {col.hi}, mid: {col.mid()}, div: {col.div()}')
    oo(data.stats(data.Cols.y, 2, "mid"))



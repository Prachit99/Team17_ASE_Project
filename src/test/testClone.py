from Constants import Constants
from Data import Data
from Utils import oo


def testClone():
    file = Constants().file
    data_1 = Data(file)
    data_2 = data_1.clone(data_1.rows)
    oo(data_1.stats('mid', data_1.cols.y, 2))
    oo(data_2.stats('mid', data_2.cols.y, 2))

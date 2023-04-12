import sys
sys.path.insert(1, "../code")
from Utils import *
import Constants
import Data


def testTree():
    data=Data.Data(Constants.Constants().file)
    showTree(data.tree(),"mid",data.cols.y,1)
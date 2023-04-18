import sys
sys.path.insert(1, "../src")
from Utils import *
import Constants
import Data


def testTree():
    data=Data.Data(Constants.Constants().file)
    showTree(data.tree(),"mid",data.cols.y,1)
import sys
sys.path.insert(1, "../src")
from Utils import oo
import Constants


def testThe():
    the = Constants.Constants().the
    return oo(the)
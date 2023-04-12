import sys
sys.path.insert(1, "../")
import Main
import Constants
from testNum import testNum
from testSym import testSym
from testThe import testThe
from testRand import testRand
from testSome import testSome
from testCsv import testCsv
from testData import testData
from testClone import testClone
from testCliffs import testCliffs
from testDist import testDist
from testHalf import testHalf
from testTree import testTree
from testSway import testSway
from testBins import testBins
from testXpln import testXpln

egs = {}

def eg(key, s, fun):
    const = Constants.Constants()
    help = const.help
    egs[key] = fun
    help += "  -g  {}\t{}\n".format(key, s)


eg("rand", "check random", testRand)
eg("sym", "check syms", testSym)
eg("num", "check nums", testNum)
eg("the", "check the", testThe)
eg("some", "demo of reservoir sampling", testSome)
eg("csv", "reading csv file",testCsv)
eg("data", "showing data sets", testData)
eg("clone", "replicate structure of a DATA", testClone)
eg("cliffs", "stats tests", testCliffs)
eg("dist", "distance test", testDist)
eg("half", "divide data in half", testHalf)
eg("tree", "show tree of clusters", testTree)
eg("sway", "optimizing", testSway)
eg("bins", "find deltas between rest and best", testBins)
eg("xpln","explore explanation sets",testXpln)

const = Constants.Constants()
help = const.help
options = dict()
Main.main(options, help, egs)
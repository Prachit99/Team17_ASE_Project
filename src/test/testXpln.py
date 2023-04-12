import sys
sys.path.insert(1, "../code")
from Utils import o
import Constants
import Data


def testXpln():
    file= Constants.Constants().file
    data = Data.Data(file)
    best,rest,evals = data.sway()
    rule,most= data.xpln(best,rest)
    print("\n-----------\nexplain=", o(data.showRule(rule)))
    temp= data.selects(rule,data.rows)
    temp2 = [r for r in temp if r!=None]
    data1= data.clone(temp2)
    print("all               ",o(data.stats('mid', data.cols.y, 2)),o(data.stats('div', data.cols.y, 2)))
    print("sway with",evals,"evals",o(best.stats('mid', best.cols.y, 2)),o(best.stats('div', best.cols.y, 2)))
    print("xpln on",evals,"evals",o(data1.stats('mid', data1.cols.y, 2)),o(data1.stats('div', data1.cols.y, 2)))
    top,_ = data.betters(len(best.rows))
    top = data.clone(top)
    print("sort with",len(data.rows),"evals",o(top.stats('mid', top.cols.y, 2)),o(top.stats('div', top.cols.y, 2)))
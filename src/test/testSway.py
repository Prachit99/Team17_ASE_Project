import sys
sys.path.insert(1, "../")
import Constants
import Data
from Utils import o


def testSway():
    data = Data.Data(Constants.Constants().file)
    best,rest,eval = data.sway()
    print("\nall ", o(data.stats('mid', data.cols.y, 2)))
    print("    ", o(data.stats('div', data.cols.y, 2)))
    print("\nbest",o(best.stats('mid', data.cols.y, 2)))
    print("    ", o(best.stats('div', data.cols.y, 2)))
    print("\nrest", o(rest.stats('mid', data.cols.y, 2)))
    print("    ", o(rest.stats('div', data.cols.y, 2)))
    # print("\nall != best?",o(diffs(best.cols.y,data.cols.y)))
    # print("best != rest?",o(diffs(best.cols.y,rest.cols.y)))


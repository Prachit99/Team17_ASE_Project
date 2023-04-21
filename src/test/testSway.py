import sys
sys.path.insert(1, "../")
import Constants
import Data
from Utils import o


def testSway():
    data = Data.Data(Constants.Constants().file)
    best,rest,eval = data.sway()
    print("\nall ", o(data.stats(data.Cols.y, 2, 'mid')))
    print("    ", o(data.stats(data.Cols.y, 2, 'div')))
    print("\nbest",o(best.stats(data.Cols.y, 2, 'mid')))
    print("    ", o(best.stats(data.Cols.y, 2, 'div')))
    print("\nrest", o(rest.stats(data.Cols.y, 2, 'mid')))
    print("    ", o(rest.stats(data.Cols.y, 2, 'div')))
    # print("\nall != best?",o(diffs(best.cols.y,data.cols.y)))
    # print("best != rest?",o(diffs(best.cols.y,rest.cols.y)))


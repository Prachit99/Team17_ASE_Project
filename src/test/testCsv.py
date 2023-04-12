import Utils
import Constants


n = 0


def no_of_chars(t):
    global n 
    n += len(t)


def testCsv():
    file = Constants.Constants().file
    Utils.csv(file,no_of_chars)
    return n == 3192

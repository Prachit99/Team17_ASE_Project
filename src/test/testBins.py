import sys
sys.path.insert(1, "../")
import Data
import Constants
from Utils import  rnd,o,value,bins

#dont know where bins function will be
#donr know if to use o function
#value in utils missing

def testBins():
    global b4
    data = Data.Data(Constants.Constants().file)
    best,rest,eval = data.sway()
    print("all","","","",o({'best':len(best.rows), 'rest':len(rest.rows)}))
    b4 = []
    for k,t in enumerate(bins(data.cols.x,{'best':best.rows, 'rest':rest.rows})):
        for range in t:
            print(range['txt'])
            if range['txt'] != b4:
                print("")
            b4 = range['txt']
            print(range['txt'],range['lo'],range['hi'],rnd(value(range['y'].has, len(best.rows),len(rest.rows),"best"))) 
            o(range['y'].has)


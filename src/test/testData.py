import Constants
import pandas as pd
import Data
from Utils import  oo
from Num import Num
from Sym import Sym


def testData():
    file = Constants.Constants().file
    data = Data.Data(file)
    df = pd.DataFrame()
    names, lows, highs, mids, divs = [], [], [], [], []
    for col in data.Cols.x:
        if type(col) == Num:
            names.append(col.txt)
            lows.append(col.lo)
            highs.append(col.hi)
            mids.append(col.mid())
            divs.append(col.div())
            # print(f'{col.txt} {type(col)}')
            # print(f'low: {col.lo}, high: {col.hi}, mid: {col.mid()}, div: {col.div()}')
        elif type(col) == Sym:
            names.append(col.txt)
            lows.append(" ")
            highs.append(" ")
            mids.append(col.mid())
            divs.append(col.div())
            # print(f'{col.txt} {type(col)}')
            # print(f'mid: {col.mid()}, div: {col.div()}')

    df['name'] = names
    df['low'] = lows
    df['high'] = highs
    df['mid'] = mids
    df['div'] = divs
    # df.to_csv("china_stats.csv")    
    oo(data.stats(data.Cols.y, 2, "mid"))



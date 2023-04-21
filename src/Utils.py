import numpy as np
import pandas as pd
import math
import random
import itertools
from operator import itemgetter
from sklearn.preprocessing import LabelEncoder
import Constants
import io
import copy
import re
import json
from Sym import Sym
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval



const = Constants.Constants()
seed = const.seed

# Utility functions for numerics
def rint(lo, hi, Seed = None):
    return math.floor(0.5 + rand(lo,hi, Seed))


def rand(lo=0, hi=1, Seed = None):
    """
    This function generates the random number between a given range
    """
    global seed
    if(Seed):
        seed = Seed
    seed = (16807 * seed) % 2147483647
    return lo + (hi - lo) * seed / 2147483647


def rnd(n, nPlaces=3):
    mult = 10**(nPlaces)
    return math.floor(n * mult + 0.5) / mult


# Utility functions for Strings
def fmt(**sControl):
    return print(sControl)


def o(t):
    if type(t) != list:
        return str(t)
    def fun(k,v):
        if not str(k).find('^_'):
            return fmt(":{} {}",o(k),o(v))


def oo(t):
    print(o(t))
    return t


def coerce(s):
    s = str(s)
    def fun(s1):
        if s1.lower() == "true":
            return True
        elif s1.lower() == "false":
            return False
        else:
            return s1
    s = fun(s)
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return fun(s)
    except Exception as exception:
        print("Coerce Error", exception)


# Utility functions for Lists
# def map(t,fun):
    # u = []
    # for k,v in enumerate(t):
    #     v,k = fun(k)
    #     index = k if k != 0 else 1+len(u)
    #     u[index] = v
    # return u


def kap(t, fun):
    u = {}
    for k,v in enumerate(t):
        # print(k, v.txt)
        v,k = fun(k,v)
        index = k if k != 0 else 1 + len(u)
        u[index] = v
    # print(u)
    return u


def sort(t, fun):
    return sorted(t, key = fun)


def keys(t):
    return sorted(kap(t,))


def csv(filename: str, fun):
    file = io.open(filename)
    t = []
    for line in file.readlines():
        row = list(map(coerce, line.strip().split(',')))
        t.append(row)
        fun(row)
    file.close()

def show(node, what, cols, nPlaces, lvl = 0):
  if node:
    print('| ' * lvl + str(len(node['data'].rows)) + '  ', end = '')
    if not node.get('left') or lvl==0:
        print(node['data'].stats("mid",node['data'].cols.y,nPlaces))
    else:
        print('')
    show(node.get('left'), what,cols, nPlaces, lvl+1)
    show(node.get('right'), what,cols,nPlaces, lvl+1)

def many(t,n):
    u=[]
    for r in range(1,n+1):
        u.append(any(t))
    return u

def any(t):
    return t[rint(0, len(t) - 1)]

def cosine(a,b,c):
    if c==0:
        d=1
    else:
        d=2*c
    x1 = (a**2 + c**2 - b**2) / d
    x2 = max(0, min(1, x1))
    y  = abs((a**2 - x2**2))**.5
    return x2, y


def repCols(cols, data):
    cols = copy.deepcopy(cols)
    for col in cols:
        col[len(col)-1] = col[0] + ":" + col[len(col)-1]
        for j in range(1, len(col)):
            col[j-1] = col[j]
        col.pop()
    col_1 = ['Num' + str(k+1) for k in range(len(cols[1])-1)]
    col_1.append('thingX')
    cols.insert(0, col_1)
    return data(cols)


def repRows(t, data, rows):
    rows = copy.deepcopy(rows)
    for j,s in enumerate(rows[-1]):
        rows[0][j] += ":" + s
    rows.pop()
    for n,row in enumerate(rows):
        if n == 0:
            row.append('thingX')
        else:
            u = t['rows'][-n]
            row.append(u[len(u)-1])
    return data(rows)


def doFile(file):
    file = open(file, 'r', encoding='utf-8')
    #print(re.findall(r'(?<=return )[^.]*', file.read())[0])
    text  = re.findall(r'(?<=return )[^.]*', file.read())[0].replace('{', '[').replace('}',']').replace('=',':').replace('[\n','{\n' ).replace(' ]',' }' ).replace('\'', '"').replace('_', '"_"')
    print(text)
    file.close()
    return json.loads(re.sub("(\w+):", r'"\1":', text))


def transpose(t):
    u=[]
    for i in range(len(t[1])):
        u.append([])
        for j in range(len(t)):
            u[i].append(t[j][i])
    return u


def repPlace(data):
    n,g = 20,{}
    for i in range(1, n+1):
        g[i] = {}
        for j in range(1, n+1):
            g[i][j] = ' '
    y_max = 0
    print('')
    for r,row in enumerate(data.rows):
        # print((data.rows))
        c = chr(97+r).upper()
        print(c, row.cells[-1])
        x,y = row.x*n//1, row.y*n//1
        y_max = int(max(y_max,y+1))
        g[y+1][x+1] = c
    print('')
    for y in range(1,y_max+1):
        print(' '.join(g[y].values()))


def repGrid(file, data):
    t = doFile(file)
    rows = repRows(t, data, transpose(t['cols']))
    cols = repCols(t['cols'], data)
    show(rows.cluster(),"mid",rows.cols.all,1)
    show(cols.cluster(),"mid",cols.cols.all,1)
    repPlace(rows)


def RANGE(at,txt,lo,hi=None):
    return {'at':at,'txt':txt,'lo':lo,'hi':lo or hi or lo,'y':Sym()}


def extend(range, n, s):
    range['lo'] = min(n, range['lo'])
    range['hi'] = max(n, range['hi'])
    range['y'].add(s)


def merge(col1, col2):
  new = copy.deepcopy(col1)
  if isinstance(col1, Sym):
      for n in col2.has:
        new.add(n)
  else:
    for n in col2.has:
        new.add(new, n)
    new.lo = min(col1.lo, col2.lo)
    new.hi = max(col1.hi, col2.hi) 
  return new


def merge2(col1, col2):
  new = merge(col1, col2)
  if new.div() <= (col1.div()*col1.n + col2.div()*col2.n)/new.n:
    return new
  

def mergeAny(ranges0):
    def noGaps(t):
        for j in range(1, len(t)):
            t[j]['lo'] = t[j - 1]['hi']
        t[0]['lo']  = -float("inf")
        t[len(t) - 1]['hi'] =  float("inf")
        return t

    ranges1 = []
    j = 0
    while j <= len(ranges0) - 1:
        left = ranges0[j]

        if j == len(ranges0)-1:
            right = None
        else: 
            right = ranges0[j+1]

        if right:
            y = merge2(left['y'], right['y'])
            if y:
                j = j+1
                left['hi'], left['y'] = right['hi'], y
        ranges1.append(left)
        j = j+1

    if len(ranges0) == len(ranges1):
        return noGaps(ranges0)
    else: 
        return mergeAny(ranges1)
    

def bins(cols, rowss):
    out = []
    for col in cols:
        ranges = {}
        for y,rows in rowss.items():
            for row in rows:
                x = row.cells[col.at]
                if x != "?":
                    k = int(bin(col, x))
                    if not k in ranges:
                        ranges[k] = RANGE(col.at, col.txt, x)
                    extend(ranges[k], x, y)
        ranges = list(dict(sorted(ranges.items())).values())
        r = ranges if isinstance(col, Sym) else mergeAny(ranges)
        out.append(r)
    return out


def bin(col, x):
    if x=="?" or isinstance(col, Sym):
        return x
    tmp = (col.hi - col.lo)/(const.bins - 1)
    return 1 if col.hi == col.lo else math.floor(x/tmp + .5) * tmp


def value(has,nB = None, nR = None, sGoal = None):
    sGoal = sGoal if sGoal else True
    nB = nB if nB else 1 
    nR = nR if nR else 1
    b,r = 0,0
    for x,n in has.items():
        if x == sGoal:
            b += n
        else:
            r += n
    b = b/(nB+1/float("inf"))
    r = r/(nR+1/float("inf"))
    return pow(b,2)/(b+r)


def showTree(node, what, cols, nPlaces, lvl = 0):
  if node:
    print('|.. ' * lvl + '[' + str(len(node['data'].rows)) + ']' + '  ', end = '')
    if not node.get('left') or lvl==0:
        print(node['data'].stats("mid",node['data'].cols.y,nPlaces))
    else:
        print('')
    showTree(node.get('left'), what,cols, nPlaces, lvl+1)
    showTree(node.get('right'), what,cols,nPlaces, lvl+1)


def prune(rule, maxSize):
    n=0
    for txt,ranges in rule.items():
        n += 1
        if len(ranges) == maxSize[txt]:
            n-=1
            rule[txt] = None
    if n > 0:
        return rule
    

def dkap(t, fun):
    u = {}
    for k,v in t.items():
        v, k = fun(k,v) 
        u[k or len(u)] = v
    return u


def firstN(sortedRanges,scoreFun):
    print("")
    def function(r):
        print(r['range']['txt'],r['range']['lo'],r['range']['hi'],rnd(r['val']),o(r['range']['y'].has))
    temp = list(map(function, sortedRanges))
    first = sortedRanges[0]['val']
    def useful(range):
        if range['val']>0.05 and range['val']> first/10:
            return range
    sortedRanges = [x for x in sortedRanges if useful(x)]
    most,out = -1, -1
    for n in range(1,len(sortedRanges)+1):
        slice = sortedRanges[0:n]
        slice_range = [x['range'] for x in slice]
        tmp,rule = scoreFun(slice_range)
        if tmp and tmp > most:
            out,most = rule,tmp
    return out,most


def better_dicts(dict1, dict2, data):
    s1, s2, ys, x, y = 0, 0, data.cols.y, None, None

    for col in ys:
        x = dict1.get(col.txt)
        y = dict2.get(col.txt)
        x = col.norm(x)
        y = col.norm(y)
        s1 = s1 - math.exp(col.w * (x - y) / len(ys))
        s2 = s2 - math.exp(col.w * (y - x) / len(ys))

    return s1 / len(ys) < s2 / len(ys)


def selects(self, rule, rows):
        def disjunction(ranges, row):
            for range in ranges:
                lo, hi, at = range['lo'], range['hi'], range['at']
                x = row.cells[at]
                if x == "?":
                    return True
                if lo == hi and lo == x:
                    return True
                if lo <= x and x < hi:
                    return True
            return False
        def conjunction(row):
            for ranges in rule.values():
                if not disjunction(ranges, row):
                    return False
            return True
        def function(r):
            if conjunction(r):
                return r
        return list(map(function, rows))


def impute_missing_values(df):
    for i in df.columns[df.isnull().any(axis=0)]:
        df[i].fillna(df[i].mean(),inplace=True)
    for col in df.columns[df.eq('?').any()]:
        df[col] =df[col].replace('?', np.nan)
        df[col] = df[col].astype(float)
        df[col] = df[col].fillna(df[col].mean())
    return df


def stats_average(data_array):
    res = {}
    for x in data_array:
        for k,v in x.stats().items():
            res[k] = res.get(k,0) + v
    for k,v in res.items():
        res[k] /= const.iter
    return res


def label_encoding(df):
    syms = [col for col in df.columns if col.strip()[0].islower() and df[col].dtypes == 'O']
    le = LabelEncoder()
    for sym in syms:
        col = le.fit_transform(df[sym])
        df[sym] = col.copy()


def preprocess_data(file, Data):
    df = pd.read_csv(file)
    df = impute_missing_values(df)
    label_encoding(df)
    file = file.replace('.csv', '_preprocessed.csv')
    df.to_csv(file, index=False)
    return Data(file)


def hpo_minimal_sampling_params(DATA, XPLN, selects, showRule):
    most_optimial_sway_hps = []
    least_sway_evals = -1
    least_xpln_evals = -1
    best_sway = ""
    best_xpln = ""
    hp_grid = const.hp_grid
    hpo_minimal_sampling_samples = const.hpo_minimal_sampling_samples
    combs = list(itertools.product(*hp_grid.values()))
    all_hp_combs = []
    for params in combs:
        all_hp_combs.append(dict(zip(hp_grid.keys(), params)))
    hp_combs_sample = random.sample(all_hp_combs, hpo_minimal_sampling_samples)
                           
    for hps in hp_combs_sample:
        const.hp.append(hps)
        data=DATA(const.file) 
        best,rest, evals = data.sway() 
        xp = XPLN(best, rest)
        rule,_= xp.xpln(data,best,rest)
        data1= DATA(data,selects(rule,data.rows))
        current_sway =  best.stats(best.cols.y, 2, 'mid')
        top,_ = data.betters(len(best.rows))
        top = data.clone(top)

        if(best_xpln == ""):
            best_xpln = data1.stats(data1.cols.y, 2, 'mid')
            best_sway = best.stats(best.cols.y, 2, 'mid')
            most_optimial_sway_hps = hps
            least_sway_evals = evals
            least_xpln_evals = evals
        else:
            if better_dicts(current_sway, best_sway, data):
                best_sway = current_sway
                most_optimial_sway_hps = hps
                least_sway_evals = evals
    const.most_optimial_sway_hps = most_optimial_sway_hps
    print('--------------- HPO Minimal Sampling Results ---------------')
    print('Best Params: ', most_optimial_sway_hps)
    print("\n-----------\nexplain=", showRule(rule))
    print("all               ",data.stats(data.cols.y, 2, 'mid'))
    print("sway with",least_sway_evals,"evals",best_sway)
    print("xpln on",least_xpln_evals,"evals",best_xpln)
    print("sort with",len(data.rows),"evals",top.stats(top.cols.y, 2, 'mid'))

def hpo_hyperopt_params(DATA, XPLN, selects, showRule):
    def hyperopt_objective(params):
        current_sway = {}
        current_xpln = {}
        top = {}
        sum = 0
        const.update(params)
        data=DATA(const.file) 
        best,rest, evals = data.sway() 
        xp = XPLN(best, rest)
        rule,_= xp.xpln(data,best,rest)
        if rule != -1:
            data1= DATA(data,selects(rule,data.rows))
            current_sway =  best.stats(best.cols.y, 2, 'mid')
            current_xpln = data1.stats(data1.cols.y, 2, 'mid')
            top,_ = data.betters(len(best.rows))
            top = data.clone(top)
            ys = data.cols.y
            for col in ys:
                x = current_sway.get(col.txt)
                sum += x * col.w
        return {"loss": sum, "status": STATUS_OK, "data": data, "evals": evals, "rule": rule, "sway": current_sway, "xpln": current_xpln, "top": top, "params": params}
    
    space = {}
    trials = Trials()
    for key in hp_grid.keys():
        space[key] = hp.choice(key, hp_grid[key])
    
    best = fmin(hyperopt_objective, space, algo=tpe.suggest, max_evals=hpo_hyperopt_samples, trials=trials)
    trial_loss = np.asarray(trials.losses(), dtype=float)
    best_ind = np.argmin(trial_loss)
    best_trial = trials.trials[best_ind]
    print('--------------- HPO Hyperopt Results ---------------')
    print('Best Params: ', best_trial['result']['params'])
    print("\n-----------\nexplain=", showRule(best_trial['result']['rule']))
    print("all               ",best_trial['result']['data'].stats(best_trial['result']['data'].cols.y, 2, 'mid'))
    print("sway with",best_trial['result']['evals'],"evals",best_trial['result']['sway'])
    print("xpln on",best_trial['result']['evals'],"evals",best_trial['result']['xpln'])
    print("sort with",len(best_trial['result']['data'].rows),"evals",best_trial['result']['top'].stats(best_trial['result']['top'].cols.y, 2, 'mid'))
    print()

def run_stats(data, top_table):
    print('MWU and KW significance level: ', const.significance_level/100)
    sway1 = top_table['sway1']
    sway2 = top_table['sway2 (kmeans)']
    # sway3 = top_table['sway3 (agglo)']
    # sway4 = top_table['sway4 (kmeans)']
    sways = ['sway1', 'sway2']
    mwu_sways = []
    kw_sways = []

    for col in data.Cols.y:
        sway_avgs = [sway1['avg'][col.txt], sway2['avg'][col.txt]]

        if col.w == -1:
            best_avg = min(sway_avgs)
        else:
            best_avg = max(sway_avgs)
        best_sway = sways[sway_avgs.index(best_avg)]

        for best in sway1['data']:
            sway1_col = [row.cells[col.at] for row in best.Rows]
        for best in sway2['data']:
            sway2_col = [row.cells[col.at] for row in best.Rows]
        # for best in sway3['data']:
        #     sway3_col = [row.cells[col.at] for row in best.Rows]
        # for best in sway4['data']:
        #     sway4_col = [row.cells[col.at] for row in best.Rows]

        # _, p_value = kruskal(sway1_col, sway2_col, sway3_col, sway4_col)
        # if p_value < the['kw_significance']:
        #     top_table['kw_significant'].append('yes')
        # else:
        #     top_table['kw_significant'].append('no')
        
        groups = [sway1_col, sway2_col]
        num_groups = len(groups)
        p_values_mwu = np.zeros((num_groups, num_groups))
        p_values_kruskal = np.zeros((num_groups, num_groups))

        for i in range(num_groups):
            for j in range(i+1, num_groups):
                _, p = mannwhitneyu(groups[i], groups[j], alternative='two-sided')
                p_values_mwu[i, j] = p
                p_values_mwu[j, i] = p
                _, p_value_kruskal = kruskal(sway1_col, sway2_col)
                p_values_kruskal[i, j] = p_value_kruskal
                p_values_kruskal[j, i] = p_value_kruskal

        # print('Pairwise Mann-Whitney U tests')
        # print(pd.DataFrame(p_values_mwu, index=sways, columns=sways))
        # print()

        # Apply Bonferroni correction for multiple comparisons
        adjusted_p_values = multipletests(p_values_mwu.ravel(), method='fdr_bh')[1].reshape(p_values_mwu.shape)
        post_hoc = pd.DataFrame(adjusted_p_values, index=sways, columns=sways)

        # print("Pairwise Mann-Whitney U tests with Benjamini/Hochberg correction:")
        # print(post_hoc)
        # print()

        krusal_df = pd.DataFrame(p_values_kruskal, index=sways, columns=sways)
        # print("Pairwise Kruskal Wallis:")
        # print(krusal_df)
        # print()

        mwu_sig = set(post_hoc.iloc[list(np.where(post_hoc >= const.significance_level)[0])].index)
        if len(mwu_sig) == 0:
            mwu_sways.append([best_sway])
        else:
            mwu_sways.append(list(mwu_sig))

        kw_sig = set(krusal_df.iloc[list(np.where(krusal_df >= const.significance_level)[0])].index)
        if len(kw_sig) == 0:
            kw_sways.append([best_sway])
        else:
            kw_sways.append(list(kw_sig))

    
    xpln1 = top_table['xpln1']
    xpln2 = top_table['xpln2 (kmeans+kdtree)']
    # xpln3 = top_table['xpln3 (agglo+kdtree)']
    # xpln4 = top_table['xpln4 (pca+kdtree)']
    xplns = ['xpln1', 'xpln2']
    mwu_xplns = []
    kw_xplns = []

    for col in data.Cols.y:
        xpln_avgs = [xpln1['avg'][col.txt], xpln2['avg'][col.txt]]

        if col.w == -1:
            best_avg = min(xpln_avgs)
        else:
            best_avg = max(xpln_avgs)
        best_xpln = xplns[xpln_avgs.index(best_avg)]

        for best in xpln1['data']:
            xpln1_col = [row.cells[col.at] for row in best.Rows]
        for best in xpln2['data']:
            xpln2_col = [row.cells[col.at] for row in best.Rows]
        # for best in xpln3['data']:
        #     xpln3_col = [row.cells[col.at] for row in best.Rows]
        # for best in xpln4['data']:
        #     xpln4_col = [row.cells[col.at] for row in best.Rows]
        
        groups = [xpln1_col, xpln2_col]
        num_groups = len(groups)
        p_values_mwu = np.zeros((num_groups, num_groups))
        p_values_kruskal = np.zeros((num_groups, num_groups))

        for i in range(num_groups):
            for j in range(i+1, num_groups):
                _, p = mannwhitneyu(groups[i], groups[j], alternative='two-sided')
                p_values_mwu[i, j] = p
                p_values_mwu[j, i] = p
                _, p_value_kruskal = kruskal(xpln1_col, xpln2_col)
                p_values_kruskal[i, j] = p_value_kruskal
                p_values_kruskal[j, i] = p_value_kruskal

        adjusted_p_values = multipletests(p_values_mwu.ravel(), method='fdr_bh')[1].reshape(p_values_mwu.shape)
        post_hoc = pd.DataFrame(adjusted_p_values, index=xplns, columns=xplns)
        krusal_df = pd.DataFrame(p_values_kruskal, index=xplns, columns=xplns)

        mwu_sig = set(post_hoc.iloc[list(np.where(post_hoc >= const.significance_level)[0])].index)
        if len(mwu_sig) == 0:
            mwu_xplns.append([best_xpln])
        else:
            mwu_xplns.append(list(mwu_sig))

        kw_sig = set(krusal_df.iloc[list(np.where(krusal_df >= const.significance_level)[0])].index)
        if len(kw_sig) == 0:
            kw_xplns.append([best_xpln])
        else:
            kw_xplns.append(list(kw_sig))

    return mwu_sways, kw_sways, mwu_xplns, kw_xplns


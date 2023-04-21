from Row import Row
from Cols import Cols
import Constants
from Utils import *
from operator import itemgetter
from functools import cmp_to_key
from sklearn.cluster import MiniBatchKMeans
import random
import numpy as np


const = Constants.Constants()
random.seed(const.seed)

class Data:
    def __init__(self, src = None, Rows = None):
        self.Rows = []
        self.Cols = None
        if src or Rows:
            if isinstance(src, str):
                csv(src, self.add)
            else:
                self.Cols = Cols(src.Cols.names)
                for Row in Rows:
                    self.add(Row)


    def add(self, t):
        if self.Cols:
            t = t if isinstance(t, Row) else Row(t)
            self.Rows.append(t)
            self.Cols.add(t)
        else:
            self.Cols=Cols(t)
    

    def stats(self, Cols = None, nPlaces = 2, what = 'mid'):
        stats_dict = dict(sorted({col.txt: rnd(getattr(col, what)(), nPlaces) for col in Cols or self.Cols.y}.items()))
        stats_dict["N"] = len(self.Rows)
        return stats_dict
    

    def dist(self, Row1, Row2, Cols = None):    
        n,d = 0,0
        for col in Cols or self.Cols.x:
            n = n + 1
            d = d + col.dist(Row1.cells[col.at], Row2.cells[col.at])**const.p
        return (d/n)**(1/const.p)


    def clone(data, init={}):
        x = Data()
        x.add(data.Cols.names)
        for _, t in enumerate(init):
            x.add(t)
        return x
  
    
    def half(self, Rows = None, Cols = None, above = None):
        def gap(Row1,Row2): 
            return self.dist(Row1,Row2,Cols)
        def project(Row):
            return {'Row' : Row, 'dist' : cosine(gap(Row,A), gap(Row,B), c)}
        Rows = Rows or self.Rows
        some = many(Rows,const.halves)
        A    = above if above and const.reuse else any(some)
        tmp = sorted([{'Row': r, 'dist': gap(r, A)} for r in some], key=lambda x: x['dist'])
        far = tmp[int((len(tmp) - 1) * const.far)]
        B    = far['Row']
        c    = far['dist']
        left, right = [], []
        for n,tmp in enumerate(sorted(map(project, Rows), key=lambda x: x['dist'])):
            if (n + 1) <= (len(Rows) / 2):
                left.append(tmp["Row"])
            else:
                right.append(tmp["Row"])
        evals = 1 if const.reuse and above else 2
        return left, right, A, B, c, evals
    

    def better(self, Rows1, Rows2, s1=0, s2=0, ys=None, x=0, y=0):
        if isinstance(Rows1, Row):
            Rows1 = [Rows1]
            Rows2 = [Rows2]
        if not ys:
            ys = self.Cols.y
        for col in ys:
            for Row1, Row2 in zip(Rows1, Rows2):
                x = col.norm(Row1.cells[col.at])
                y = col.norm(Row2.cells[col.at])
                s1 = s1 - (pow(math.e,(col.w*(x-y))/len(ys)))
                s2 = s2 - (pow(math.e,(col.w*(y-x))/len(ys)))
        return s1 / len(ys) < s2 / len(ys)

    
    def tree(self, Rows = None , min = None, Cols = None, above = None):
        Rows = Rows or self.Rows
        min  = min or len(Rows)**const.min
        Cols = Cols or self.Cols.x
        node = { 'Data' : self.clone(Rows) }
        if len(Rows) >= 2*min:
            left, right, node['A'], node['B'], _, _ = self.half(Rows,Cols,above)
            node['left']  = self.tree(left,  min, Cols, node['A'])
            node['right'] = self.tree(right, min, Cols, node['B'])
        return node
    
    def sway(self, algo='half'):
        data = self
        def worker(Rows, worse, evals0 = None, above = None):
            if len(Rows) <= len(data.Rows)**const.min: 
                return Rows, many(worse, const.rest*len(Rows)), evals0
            else:
                if algo == 'half':
                    l,r,A,B,c,evals = self.half(Rows, None, above)
                elif algo == 'min_batch_kmeans':
                    l,r,A,B,evals = self.mini_kmeans(Rows)
                
                if self.better(B,A):
                    l,r,A,B = r,l,B,A
                
                for Row in r:
                    worse.append(Row)
                return worker(l,worse,evals+evals0,A)
        best,rest,evals = worker(data.Rows,[],0)
        return Data.clone(self, best), Data.clone(self, rest), evals
    
    def betters(self,n):
        key = cmp_to_key(lambda Row1, Row2: -1 if self.better(Row1, Row2) else 1)
        tmp = sorted(self.Rows, key = key)
        if n is None:
            return tmp
        else:
            return tmp[1:n], tmp[n+1:]
    
    
    def mini_kmeans(self, Rows=None):
        left = []
        right = []
        A = None
        B = None
        
        def min_dist(center, Row, A):
            if not A:
                A = Row
            if self.dist(A, center) > self.dist(A, Row):
                return Row
            else:
                return A
    
        if not Rows:
            Rows = self.Rows
        Row_set = np.array([r.cells for r in Rows])
        kmeans = MiniBatchKMeans(n_clusters=2, random_state=0, batch_size=100, n_init="auto")
        kmeans.fit(Row_set)
        left_cluster = Row(kmeans.cluster_centers_[0])
        right_cluster = Row(kmeans.cluster_centers_[1])

        for key, value in enumerate(kmeans.labels_):
            if value == 0:
                A = min_dist(left_cluster, Rows[key], A)
                left.append(Rows[key])
            else:
                B = min_dist(right_cluster, Rows[key], B)
                right.append(Rows[key])

        return left, right, A, B, 1
    
    def xpln(self,best,rest):
        tmp, maxSizes = [], {}
        def v(has):
            return value(has, len(best.Rows), len(rest.Rows), "best")
        def score(ranges):
            rule = self.rule(ranges,maxSizes)
            if rule:
                print(self.showRule(rule))
                bestr= self.selects(rule, best.Rows)
                restr= self.selects(rule, rest.Rows)
                if len(bestr) + len(restr) > 0: 
                    return v({'best': len(bestr), 'rest':len(restr)}), rule
            return None, None
    
        # def score(self, ranges):
        #     rule = self.RULE(ranges, self.maxSizes)
        #     if rule:
        #         bestr = selects(rule, self.best.Rows)
        #         restr = selects(rule, self.rest.Rows)
        #         if len(bestr) + len(restr) > 0:
        #             return value({'best': len(bestr), 'rest': len(restr)}, len(self.best.Rows), len(self.rest.Rows), 'best'), rule
        #     return None,None

        for ranges in bins(self.Cols.x,{'best':best.Rows, 'rest':rest.Rows}):
            maxSizes[ranges[0]['txt']] = len(ranges)
            print("")
            for range in ranges:
                print(range['txt'], range['lo'], range['hi'])
                tmp.append({'range':range, 'max':len(ranges),'val': v(range['y'].has)})

        rule,most=self.firstN(sorted(tmp, key=itemgetter('val')),score)
        return rule,most
    
    def firstN(self, sorted_ranges, scoreFun):
        first = sorted_ranges[0]['val']

        def useful(range):
            if range['val'] > 0.05 and range['val'] > first / 10:
                return range
        sorted_ranges = [s for s in sorted_ranges if useful(s)]
        most = -1
        out = -1
        for n in range(len(sorted_ranges)):
            tmp, rule = scoreFun([r['range'] for r in sorted_ranges[:n+1]])
            if tmp is not None and tmp > most:
                out, most = rule, tmp
        return out, most
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

        # def function(r):
        #     if conjunction(r):
        #         return r
        fun = lambda r: r if conjunction(r) else None

        # return list(map(function, rows))
        return list(map(fun, rows))
    
    def rule(self,ranges,maxSize):
        t={}
        for range in ranges:
            t[range['txt']] = t.get(range['txt']) if t.get(range['txt']) else []
            t[range['txt']].append({'lo' : range['lo'],'hi' : range['hi'],'at':range['at']})
        return prune(t, maxSize)
        
    def showRule(self,rule):
        def pretty(range):
            return range['lo'] if range['lo']==range['hi'] else [range['lo'], range['hi']]
    
        def merge(t0):
            t,j =[],1
            while j<=len(t0):
                left = t0[j-1]
                if j < len(t0):
                    right = t0[j]
                else:
                    right = None
                if right and left['hi']==right['lo']:
                    left['hi']=right['hi']
                    j+=1
                t.append({'lo':left['lo'], 'hi':left['hi']})
                j+=1

            return t if len(t0)==len(t) else merge(t) 
        def merges(attr,ranges):
            print("Ranges")
            print(ranges)
            return list(map(pretty,merge(sorted(ranges,key=itemgetter('lo'))))),attr
        return dkap(rule,merges)
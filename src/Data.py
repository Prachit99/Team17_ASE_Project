from Row import Row
from Cols import Cols
import Constants
from Utils import *
from operator import itemgetter
from functools import cmp_to_key
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MiniBatchKMeans
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

    def clone(data, ts={}):
        Data1 = Data()
        Data1.add(data.Cols.names)
        for _, t in enumerate(ts or {}):
            Data1.add(t)
        return Data1

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
                # print(col.at,col.txt,"Data Line 84")
                x = col.norm(Row1.cells[col.at])
                y = col.norm(Row2.cells[col.at])
                s1 = s1 - math.exp(col.w * (x - y) / len(ys))
                s2 = s2 - math.exp(col.w * (y - x) / len(ys))
        return s1 / len(ys) < s2 / len(ys)
    
    def bdom(self, Rows1, Rows2, ys=None):
        if isinstance(Rows1, Row):
            Rows1 = [Rows1]
            Rows2 = [Rows2]
        if not ys:
            ys = self.Cols.y
        
        dominates = False
        for col in ys:
            for Row1, Row2 in zip(Rows1, Rows2):
                x = col.norm(Row1.cells[col.at]) * col.w * -1
                y = col.norm(Row2.cells[col.at]) * col.w * -1
                if x > y:
                    return False
                elif x < y:
                    dominates = True
        return dominates

    def better_bdom(self, Row1, Row2, ys=None):
        Row1_bdom = self.bdom(Row1, Row2, ys=ys)
        Row2_bdom = self.bdom(Row2, Row1, ys=ys)
        if Row1_bdom and not Row2_bdom:
            return True
        else:
            return False
        
    def better_hypervolume(self, Row1, Row2):
        s1, s2, ys = 0, 0, self.Cols.y
        Data = [[], []]
        ref_point = []
        for col in ys:
            x = col.norm(Row1.cells[col.at])
            y = col.norm(Row2.cells[col.at])
            if '-' in col.txt:
                x = -x
                y = -y
                ref_point.append(1)
            else:
                ref_point.append(2)
            Data[0].append(x)
            Data[1].append(y)
        if len(ref_point) < 2:
            return Data[0] > Data[1]
        # print(Data)
        # print(ref_point)

        hv = hypervolume(Data)
        output = hv.contributions(ref_point)
        hv1, hv2 = output[0], output[1]

        return hv1 < hv2
    
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
    
    def sway(self, algo = 'half', better = 'zitler'):
        data = self
        def worker(Rows, worse, evals0 = None, above = None):
            if len(Rows) <= len(data.Rows)**const.min: 
                return Rows, many(worse, const.rest*len(Rows)), evals0
            else:
                if algo == 'half':
                    l,r,A,B,c,evals = self.half(Rows, None, above)
                elif algo == 'kmeans':
                    l,r,A,B,evals = self.kmeans(Rows)
                elif algo == 'agglomerative_clustering':
                    l,r,A,B,evals = self.agglomerative_clustering(Rows)
                elif algo == 'dbscan':
                    l,r,A,B,evals = self.dbscan(Rows)
                elif algo == 'pca':
                    l,r,A,B,evals = self.pca(Rows)
                
                if better == 'zitler':
                    if self.better(B,A):
                        l,r,A,B = r,l,B,A
                elif better == 'bdom':
                    if self.better_bdom(B,A):
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
    
    def kmeans1(self, Rows=None):
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
        kmeans = KMeans(n_clusters=2, random_state=const.seed, n_init=10)
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
    
    def kmeans(self, Rows=None):
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
        kmeans = MiniBatchKMeans(n_clusters=2, random_state=0, batch_size=250, n_init=10)
        # print(Row_set,"data Line 241")
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
    

    def agglomerative_clustering(self, Rows=None):
        left = []
        right = []

        if not Rows:
            Rows = self.Rows
        Row_set = np.array([r.cells for r in Rows])
        agg_clust = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='ward')
        agg_clust.fit(Row_set)

        for key, value in enumerate(agg_clust.labels_):
            if value == 0:
                left.append(Rows[key])
            else:
                right.append(Rows[key])
        return left, right, random.choices(left, k=10), random.choices(right, k=10), 1
    
    def dbscan(self, Rows=None):
        left = []
        right = []

        if not Rows:
            Rows = self.Rows
        Row_set = np.array([r.cells for r in Rows])
        db = DBSCAN(eps = 3, min_samples = 2)
        db.fit(Row_set)

        for key, value in enumerate(db.labels_):
            if value == 0:
                left.append(Rows[key])
            else:
                right.append(Rows[key])
        return left, right, random.choices(left, k=10), random.choices(right, k=10), db.n_features_in_
    
    def pca(self, Rows=None, Cols=None, above=None):
        if not Rows:
            Rows = self.Rows
        Row_set = np.array([r.cells for r in Rows])
        pca = PCA(n_components=1)
        pcs = pca.fit_transform(Row_set)
        result = []
        for i in sorted(enumerate(Rows), key=lambda x: pcs[x[0]]):
            result.append(i[1])
        n = len(result)
        left = result[:n//2]
        right = result[n//2:]
        return left, right, random.choices(left, k=10), random.choices(right, k=10), 1
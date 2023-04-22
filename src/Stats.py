import math, random
from Num import Num
from Utils import *


def samples(t,n=0):
    u= []
    n = n or len(t)
    for i in range(n): 
        u.append(t[random.randrange(len(t))]) 
    return u


def gaussian(mu = 0, sd = 1):
    return mu + sd * math.sqrt(-2 * math.log(random())) * math.cos(2 * math.pi * random())


def cliffsDelta(ns1, ns2):
    n, gt, lt = 0,0,0
    for x in ns1:
        for y in ns2:
            n +=  1
            if x > y:
                gt += 1
            if x < y:
                lt += 1
    return abs(lt - gt)/n > const.cliffs


def delta(i, other):
    e, y, z= 1E-32, i, other
    try:
        return abs(y.mu - z.mu) / (math.sqrt(e + y.sd**2/y.n + z.sd**2/z.n))
    except ZeroDivisionError:
        return float('inf')
    

def bootstrap(y0, z0):
    x, y, z, yhat, zhat = Num(), Num(), Num(), [], []
    for y1 in y0:
        x.add(y1)
        y.add(y1)
    for z1 in z0:
        x.add(z1)
        z.add(z1)
    xmu, ymu, zmu = x.mu, y.mu, z.mu
    for y1 in y0:
        yhat.append(y1 - ymu + xmu)
    for z1 in z0:
        zhat.append(z1 - zmu + xmu)
    tobs = delta(y, z)
    n = 0
    for _ in range(const.bootstrap):

        if delta(Num(t=samples(yhat)), Num(t=samples(zhat))) > tobs:
            n += 1
    return n / const.bootstrap >= const.conf


def RX(t,s):
    name = s if s else ""
    has = sorted(t) if t else []
    return {'name': name, 'rank': 0, 'has': has, 'show':""}


def div(t):
    t= t.get('has', t)
    return (t[ len(t)*9//10 ] - t[ len(t)*1//10 ])/2.56


def mid(t):
  t = t['has'] if t['has'] else t
  n = (len(t)-1)//2
  return (t[n] +t[n+1])/2 if len(t)%2==0 else t[n+1]


def merge(rx1, rx2):
    rx3 = RX([], rx1['name'])
    rx3['has'] = rx1['has'] + rx2['has']
    rx3['has'] = sorted(rx3['has'])
    rx3['n'] = len(rx3['has'])
    return rx3
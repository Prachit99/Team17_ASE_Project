import math
from Constants import Constants

class Num:
    def __init__(self, at = 0, txt = ""):
        self.at = at
        self.txt = txt
        self.n = 0
        self.mu = 0
        self.m2 = 0
        self.lo = math.inf
        self.hi = -math.inf
        self.w = -1 if txt.endswith("-") else 1
        self.has= {}


    # Method to add 
    def add(self, n):
        if n != "?":
            self.n += 1
            if self.n <= Constants().max:
                self.has[n]= n
            d = n - self.mu
            self.mu = self.mu + d/self.n
            self.m2 = self.m2 + d*(n - self.mu)
            self.lo = min(n, self.lo)
            self.hi = max(n, self.hi)


    # Method to return the mean of Num class
    def mid(self):
        return self.mu 


    # Method to return the standard deviation
    def div(self):
        return 0 if (self.m2 < 0 or self.n < 2) else (self.m2/(self.n-1))**(0.5)

    
    # Method to return a number after rounding it 
    def rnd(self, x, n):
        return x if x == "?" else round(x, n)


    # Method to calculate the normalized value of a Num
    def norm(self, n):
        return n if n == "?" else (n - self.lo)/(self.hi - self.lo)


    # Method to calculate the distance between two Nums
    def dist(self, n1, n2):
        if n1 == "?" and n2 == "?":
            return 1
        n1, n2 = self.norm(n1), self.norm(n2)
        if n1 == "?":
            n1 = 1 if n2 < 0.5 else 0
        if n2 == "?":
            n2 = 1 if n1 < 0.5 else 0
        return abs(n1 - n2)
    

# nums = Num()
# for num in [1,2,3,4,5,6,7,8,9]:
#     nums.add(num)

# print(f'Mean: {nums.mid()}')
# print(f'STD: {nums.div()}')
# print(f'rnd: {nums.rnd(2.9034567, 4)}')
# print(f'norm: {nums.norm(8)}')
# print(f'dist: {nums.dist(8, "?")}')
import math

class Sym:
    # Constructor for Sym class
    def __init__(self, at:int=0, txt:str=""):
        self.at = at
        self.txt = txt
        self.n = 0
        self.has = {}
        self.most = 0
        self.mode = None
    

    # Method to add new symbol of class Sym
    def add(self, x: str):
        if x != "?":
            self.n += 1
            self.has[x] = (self.has[x] + 1) if x in self.has else 1
            if self.has[x] > self.most:
                self.most = self.has[x]
                self.mode = x  
    
    
    # Method to retrun the mode of Sym class 
    def mid(self) -> str:
        return self.mode


    # Method to return the entropy of the class Sym
    def div(self) -> float:
        def std_form(p):
            return p * math.log(p, 2)
        
        e = 0
        for key,val in self.has.items():
            e += std_form(val/self.n)
        return -e
    

    # Method to return a rounded string
    def rnd(self, x:str, nPlaces: int) -> str:
        return x


    # This function calculates the distance between two Sym
    def dist(self, s1: str, s2: str):
        return 1 if s1 == "?" and s2 == "?" else (0 if s1 == s2 else 1)


# syms = Sym()
# for sym in "aaaabbc":
#     syms.add(sym)

# print(f'Mean: {syms.mid()}')
# print(f'Entropy: {syms.div()}')
# print(f'rnd: {syms.rnd("a")}')
# print(f'dist: {syms.dist("a", "b")}')
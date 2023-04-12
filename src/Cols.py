import re
from Num import Num
from Sym import Sym


class Cols:
    def __init__(self,t):
        self.names=t
        self.all=[]
        self.x=[]
        self.y=[]
        self.klass=None
        for n,s in enumerate(t):
            s=str(s)
            col = Num(n,s) if re.findall("^[A-Z]+",s) else Sym(n,s)
            self.all.append(col)
            if not re.findall("X$",s):
                if re.findall("!$",s):
                    self.klass=col
                if re.findall("[!+-]$",s):
                    self.y.append(col)
                else:
                    self.x.append(col)


    def add(self, row):
        for t in [self.x,self.y]:
            for col in t:
                col.add(row.cells[col.at])
            


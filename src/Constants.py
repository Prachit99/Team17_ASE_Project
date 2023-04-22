import numpy as np

class Constants:
    def __init__(self):
        self.the = dict()
        self.iter = 20
        self.seed = 937162211
        self.bootstrap = 512
        self.p = 2
        self.cliffs=0.147
        self.sample = 512
        self.far = 0.95
        self.min = 0.5
        self.max = 512
        self.bins = 16
        self.rest = 4
        self.halves = 512
        self.conf = 0.9
        self.cliffs = 0.4
        self.cohen = 0.35
        self.Fmt = "{:.2f}"
        self.significance_level = 5
        self.d = 0.35
        self.reuse = True
        self.file = "../../etc/data/auto93.csv"
        self.help = '''
            script.lua : an example script with help text and a test suite
            (c)2022, Tim Menzies <timm@ieee.org>, BSD-2 
            USAGE:   script.lua  [OPTIONS] [-g ACTION]
            OPTIONS:
            -d  --dump  on crash, dump stack = false
            -g  --go    start-up action      = data
            -h  --help  show help            = false
            -s  --seed  random number seed   = 937162211
            ACTIONS:
        '''
        self.top_table = {'all': {'data' : [], 'evals' : 0},
             'sway1': {'data' : [], 'evals' : 0},
             'sway2': {'data' : [], 'evals' : 0},
             'xpln': {'data' : [], 'evals' : 0},
             'top': {'data' : [], 'evals' : 0}}

        self.bottom_table = [[['all', 'all'],None],
                [['all', 'sway1'],None],
                [['sway1', 'sway2'],None],
                [['sway1', 'xpln'],None],
                [['sway1', 'top'],None]]
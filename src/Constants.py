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
        self.hp = []
        self.hpo_hyperopt_samples = 100
        self.hpo_minimal_sampling_samples = 50
        self.most_optimial_sway_hps = None
        self.cohen = 0.35
        self.Fmt = "{:.2f}"
        self.significance_level = 5
        self.d = 0.35
        self.reuse = True
        self.file = "../../etc/data/SSN.csv"
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
             'xpln1': {'data' : [], 'evals' : 0},
             'xpln2': {'data' : [], 'evals' : 0},
             'top': {'data' : [], 'evals' : 0}}

        self.bottom_table = [[['all', 'all'],None],
                [['all', 'sway1'],None],
                [['sway1', 'sway2'],None],
                [['sway1', 'xpln1'],None],
                [['sway2', 'xpln2'],None],
                [['sway1', 'top'],None]]
         
        self.hp_grid = {
            'bins': [round(i, 3) for i in list(np.arange(2, 15, 2))],
            'better': ['zitler'],
            'Far': [round(i, 3) for i in list(np.arange(0.5, 1, 0.05))],
            'min': [round(i, 3) for i in list(np.arange(0.5, 1, 0.05))],
            'Max': [round(i, 3) for i in list(np.arange(12, 3000, 500))],
            'p': [round(i, 3) for i in list(np.arange(0.5, 3, 0.25))],
            'rest': [round(i, 3) for i in list(np.arange(1, 10, 1))]
        }

import re
import sys
from Utils import *
from Stats import *
import Constants
from Data import Data 
from tabulate import tabulate
from xpln import *


def settings(s):
    regexp = "\n[\s]+[-][\S]+[\s]+[-][-]([\S]+)[^\n]+= ([\S]+)"
    value = re.findall(regexp, s)
    return value


def cli(s):
    const = Constants.Constants()
    value = settings(s)
    for k, v in value:
        const.the[k] = coerce(v)
    the = const.the

    args = sys.argv
    args = args[1:]
    keys = the.keys()
    for key in keys:
        val = str(the[key])
        for n,x in enumerate(args):
            if x == "-"+key[0] or x == "--"+key:
                val = "False" if val == "True" else "True" if val == "False" else args[n+1]
        the[key] = coerce(val)
    return the


def main(options,help,funs):
    saved = dict()
    const = Constants.Constants()
    top_table = const.top_table
    bottom_table = const.bottom_table


    fails = 0
    passes = 0

    for k,v in cli(help).items():
        options[k] = v
        saved[k] = v

    if options['help']:
        print(help)

    else:
        count = 0
        while count < const.iter:
            data = Data(const.file)

            best, rest, evals = data.sway()
            # xp = XPLN(best, rest)
            rule,_= data.xpln(best,rest)

            best2, rest2, evals2 = data.sway('min_batch_kmeans')
            # xp2 = XPLN(best2, rest2)
            best_xpln2, rest_xpln2 = data.xpln(best2, rest2)


            if rule != -1:
                betters, _ = data.betters(len(best.Rows))
                top_table['top']['data'].append(Data(data,betters))
                top_table['xpln']['data'].append(Data(data,selects(rule,data.Rows)))
                top_table['all']['data'].append(data)
                top_table['sway1']['data'].append(best)
                top_table['sway2']['data'].append(best2)
                top_table['all']['evals'] += 0
                top_table['sway1']['evals'] += evals
                top_table['sway2']['evals'] += evals2
                top_table['xpln']['evals'] += evals
                top_table['top']['evals'] += len(data.Rows)
                
                for i in range(len(bottom_table)):
                    [base, diff], result = bottom_table[i]
                    if result == None:
                        bottom_table[i][1] = ['=' for _ in range(len(data.Cols.y))]
                    for k in range(len(data.Cols.y)):
                        if bottom_table[i][1][k] == '=':
                            y0, z0 = top_table[base]['data'][count].Cols.y[k],top_table[diff]['data'][count].Cols.y[k]
                            is_equal = bootstrap(y0.vals(), z0.vals()) and cliffsDelta(y0.vals(), z0.vals())
                            if not is_equal:
                                bottom_table[i][1][k] = '≠'
                count += 1
        
        headers = [y.txt for y in data.Cols.y]
        table = []

        top_table['sway1'] = top_table.pop('sway1')
        top_table['sway2 (mini batch kmeans)'] = top_table.pop('sway2')
        top_table['xpln'] = top_table.pop('xpln')
        top_table['top'] = top_table.pop('top')

        for k,v in top_table.items():
            v['avg'] = stats_average(v['data'])
            stats = [k] + [v['avg'][y] for y in headers]
            stats += [int(v['evals']/const.iter)]
            print("-----------------")
            print(type(stats))
            table.append(stats)
        
        print(tabulate(table, headers=headers+["n_evals avg"],numalign="right"))
        print()

        table=[]
        for [base, diff], result in bottom_table:
            table.append([f"{base} to {diff}"] + result)
        print(tabulate(table, headers = headers, numalign = "right"))

        for what, fun in funs.items():
            if options['go'] == 'all' or what == options['go']:
                for k, v in saved.items():
                    options[k] = v
                if funs[what]() == False:
                    fails += 1
                    print("❌ fail:", what)
                else:
                    passes += 1
                    print("✅ pass:", what)
    
    if passes + fails > 0:
        print({'pass' : passes, 'fail' : fails, 'success' :100*passes/(passes+fails)//1})
    # sys.exit(n)
    exit(fails)

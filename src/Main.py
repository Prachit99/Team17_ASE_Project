import re
import sys
from Utils import *
from Stats import *
import Constants
from Data import Data 
from tabulate import tabulate
from xpln import *
from xpln2 import *


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
            data2 = preprocess_data(const.file, Data)

            best2,rest2,evals2 = data2.sway('kmeans')
            xp2 = XPLN2(best2, rest2)
            best_xpln2, rest_xpln2 = xp2.decision_tree(data2)

            # best3,rest3,evals3 = data2.sway('agglomerative_clustering')
            # xp3 = XPLN2(best3, rest3)
            # best_xpln3, rest_xpln3 = xp3.decision_tree(data2)

            # best4,rest4,evals4 = data2.sway('pca')
            # xp4 = XPLN2(best4, rest4)
            # best_xpln4, rest_xpln4 = xp4.decision_tree(data2)


            best,rest,evals = data.sway()
            xp = XPLN(best, rest)
            rule,_= xp.xpln(data,best,rest)

            if rule != -1:
                betters, _ = data.betters(len(best.Rows))
                top_table['top']['data'].append(Data(data,betters))
                top_table['xpln1']['data'].append(Data(data,selects(rule,data.Rows)))
                top_table['xpln2']['data'].append(Data(data2,best_xpln2))
                # top_table['xpln3']['data'].append(Data(data2,best_xpln3))
                # top_table['xpln4']['data'].append(Data(data2,best_xpln4))
                top_table['all']['data'].append(data)
                top_table['sway1']['data'].append(best)
                top_table['sway2']['data'].append(best2)
                # top_table['sway3']['data'].append(best3)
                # top_table['sway4']['data'].append(best4)
                top_table['all']['evals'] += 0
                top_table['sway1']['evals'] += evals
                top_table['sway2']['evals'] += evals2
                # top_table['sway3']['evals'] += evals3
                # top_table['sway4']['evals'] += evals4
                top_table['xpln1']['evals'] += evals
                top_table['xpln2']['evals'] += evals2
                # top_table['xpln3']['evals'] += evals3
                # top_table['xpln4']['evals'] += evals4
                top_table['top']['evals'] += len(data.Rows)
                
                for i in range(len(bottom_table)):
                    [base, diff], result = bottom_table[i]
                    if result == None:
                        bottom_table[i][1] = ['=' for _ in range(len(data.Cols.y))]
                    for k in range(len(data.Cols.y)):
                        if bottom_table[i][1][k] == '=':
                            # print(count, k)
                            # print(top_table[base]['data'][count].Cols.y[k].txt, top_table[diff]['data'][count].Cols.y[k].txt)
                            y0, z0 = top_table[base]['data'][count].Cols.y[k],top_table[diff]['data'][count].Cols.y[k]
                            is_equal = bootstrap(y0.vals(), z0.vals()) and cliffsDelta(y0.vals(), z0.vals())
                            if not is_equal:
                                bottom_table[i][1][k] = '≠'
                count += 1
        
        # with open(const.file.replace('/data', '/out').replace('.csv', '.out'), 'w') as outfile:
        headers = [y.txt for y in data.Cols.y]
        table = []

        top_table['sway1'] = top_table.pop('sway1')
        top_table['sway2 (kmeans)'] = top_table.pop('sway2')
        # top_table['sway3 (agglo)'] = top_table.pop('sway3')
        # top_table['sway4 (pca)'] = top_table.pop('sway4')
        top_table['xpln1'] = top_table.pop('xpln1')
        top_table['xpln2 (kmeans+kdtree)'] = top_table.pop('xpln2')
        # top_table['xpln3 (agglo+kdtree)'] = top_table.pop('xpln3')
        # top_table['xpln4 (pca+kdtree)'] = top_table.pop('xpln4')
        top_table['top'] = top_table.pop('top')

        for k,v in top_table.items():
            # print(v['data'],"Main 129")
            v['avg'] = stats_average(v['data'])
            stats = [k] + [v['avg'][y] for y in headers]
            stats += [int(v['evals']/const.iter)]
            print("-----------------")
            print(type(stats))
            table.append(stats)
        
        mwu_sways, kw_sways, mwu_xplns, kw_xplns = run_stats(data2, top_table)
        table.append(['Mann-Whitney U Sways'] + mwu_sways)
        table.append(['Kruskal-Wallis Sways'] + kw_sways)
        table.append(['Mann-Whitney U Xplns'] + mwu_xplns)
        table.append(['Kruskal-Wallis Xplns'] + kw_xplns)
        
        print(tabulate(table, headers=headers+["n_evals avg"],numalign="right"))
        print()
        # outfile.write(tabulate(table, headers=headers+["n_evals avg"],numalign="right"))
        # outfile.write('\n')

        table=[]
        for [base, diff], result in bottom_table:
            table.append([f"{base} to {diff}"] + result)
        print(tabulate(table, headers = headers, numalign = "right"))
        # outfile.write(tabulate(table, headers=headers, numalign="right"))

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
        print("🔆",{'pass' : passes, 'fail' : fails, 'success' :100*passes/(passes+fails)//1})
    # sys.exit(n)
    exit(fails)

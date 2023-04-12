import re
import sys
from Utils import coerce
import Constants


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
    fails = 0
    for k,v in cli(help).items():
        options[k] = v
        saved[k] = v

    if options['help']:
        print(help)

    else:
        for what,fun, in funs.items():
            if options['go'] == 'all' or what == options['go']:
                for k, v in saved.items():
                    options[k] = v
                # seed = options['seed']
                if funs[what]() == False:
                    fails += 1
                    print("❌ fail:", what)
                else:
                    print("✅ pass:", what)

    exit(fails)

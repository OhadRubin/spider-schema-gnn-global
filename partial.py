#! /usr/bin/python3

import json
import os
import argparse
import glob
from pprint import pprint
import _jsonnet
import shutil
from jsondiff import diff

def sanitize(inp):
    if len(list(inp.keys()))>0 and list(inp.keys())[0]=="$replace":
        return sanitize(inp["$replace"])
    curr={}
    for key,value in inp.items():
        if isinstance(value, dict):
            curr[key] = sanitize(value)
        else:
            curr[key]=value
    return curr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--name')
    group.add_argument('--reset', action='store_true')
    parser.add_argument('--curr_file',default='train_configs/current.jsonnet')
    parser.add_argument('--default_file',default='train_configs/defaults.jsonnet')
    args = parser.parse_args()
    
    if args.reset:
        os.remove(args.curr_file)
        shutil.copyfile(args.default_file, args.curr_file)
    else:            
        curr_config= json.loads(_jsonnet.evaluate_file(args.curr_file,ext_vars={"gpu":"0","experiment_name":''}))
        default_config  = json.loads(_jsonnet.evaluate_file(args.default_file,ext_vars={"gpu":"0","experiment_name":''}))
        res = diff(default_config,curr_config, dump=True)
        res = sanitize(json.loads(res)) #remove pesky $replace
        with open(f"train_configs/partials/{args.name}.json","+w") as f:
            print("Resulting partial config:")
            print(json.dumps(res,indent=4))
            json.dump(res,f,indent=4) #stringify
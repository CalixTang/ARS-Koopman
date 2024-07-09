import argparse
import itertools
import os
"""
A very simple grid sweep function. I wrote it this way to make it generalizable to all scripts and param setups in this repo.
"""

# https://stackoverflow.com/a/40623158  
def dict_product(dicts):
    """
    list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))

def delimit_task_id(task_id):
    return task_id.split('-')[0]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--script_name', type = str, required = True) #need to be as if you're running it, e.g. ARS/ARS/ars.py
    parser.add_argument('--params_path', type = str, required = True)


    args = parser.parse_args()

    if args.params_path is not None:
        import json
        params = dict(json.load(open(args.params_path, 'r')))
        print(params)
    
    ctr = 0

    hyp_sweep_params = list(dict_product(params))
    print(hyp_sweep_params)


    exec_string_base = "python " + args.script_name
    
    # simple grid search for now
    for param_dict in hyp_sweep_params:
        task_name = delimit_task_id(param_dict['task_id'])
        exec_string = exec_string_base[:]

        #add options
        for k, v in param_dict.items():
            if isinstance(v, list):
                exec_string += f' --{k}'
                for elem in v:
                    exec_string += f' {elem}'
            else:
                exec_string += f' --{k} {v}'
        
        #give each run a unique name and write output to a unique report file
        exec_string += f' --run_name {task_name}-hsweeprun-{ctr} > reports/{task_name}-hsweeprun-{ctr}.out'
        print(exec_string)
        os.system(exec_string) #probably not the safest, but i hope nobody screws this up

        ctr += 1
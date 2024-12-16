import os
from pathlib import Path
import stat

from eval_main import get_previous_ckpts, get_run_ckpts

filename = 'recompute.bash'

method_first = False


devices = [0, 1, 2]

nc_methods = ['nc_train', 'nc_eval']
odd_methods = ['msp', 'odin', 'mds', 'react', 'dice', 'knn', 'nusa', 'vim', 'ncscore', 'neco', 'epa']
methods = nc_methods + odd_methods

# ckpts = get_previous_ckpts()

# run_dir = Path('/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet/ResNet50/no_noise/150+_epochs/run_e150_2024_11_12-21_40_57')
# ckpts = get_run_ckpts(run_dir)

top_dir = Path('/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/')
run_dirs = (d for d in top_dir.iterdir() if d.is_dir())
ckpts = [c for r in run_dirs for c in get_run_ckpts(r)]


if method_first:
    combinations = [(str(c), m) for m in methods for c in ckpts][::-1]
else:
    combinations = [(str(c), m) for c in ckpts for m in methods][::-1]


start = '''#!/bin/bash

cleanup() {
  pkill -P $$
  wait
  exit
}

trap cleanup SIGINT SIGTERM\n
'''

template = "krenew -- sh -c 'CUDA_VISIBLE_DEVICES={device} python recompute.py ckpt={ckpt} method={method}' &\n"

delimiter = 'wait $(jobs -p)\n\n'


with open(filename, 'w') as f:
    f.write(start)

    while combinations:
        for d in devices:
            
            try:
                c, m = combinations.pop()
            except IndexError:
                continue

            f.write(template.format(device=d, ckpt=c, method=m))
        
        f.write(delimiter)

# Make executeable
st = os.stat(filename)
os.chmod(filename, st.st_mode | stat.S_IEXEC)
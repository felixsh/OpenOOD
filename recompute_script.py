import os
import stat
from pathlib import Path

from natsort import natsorted

from eval_main import get_run_ckpts
from plot_utils import benchmark2ckptdirs, check_h5file
from utils import ckpt_to_h5file_path

filename = 'run_imagenet200.bash'
# script = 'compute_acc_train.py'
script = 'recompute.py'

with_methods = True
method_first = True
reverse = True
missing = True
write_list = True

devices = [0, 1, 2, 3]

acc_method = ['acc']
nc_method = ['nc']
odd_methods = [
    'msp',
    'odin',
    'mds',
    'react',
    'dice',
    'knn',
    'nusa',
    'vim',
    'ncscore',
    'neco',
    'epa',
]
methods = nc_method + odd_methods + acc_method

# ckpts = get_previous_ckpts()

# run_dir = Path(
#     '/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_47'
# )
# ckpts = get_run_ckpts(run_dir)
# ckpts = [c for c in ckpts if '200' in str(c) or '500' in str(c)]
# print(ckpts)

top_dir = Path(
    '/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/type/no_noise/1000+_epochs/'
)
#
# top_dir = Path('/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/NCLessNet18/no_noise/1000+_epochs/')
# run_dirs = (d for d in top_dir.iterdir() if d.is_dir())
# run_dirs = (
#     Path('/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58'),
#     Path('/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56'),
#     Path('/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14'),
#     Path('/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25'),
#     Path('/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04'),
# )

# benchmarks = [
#    'cifar100',
#    'imagenet200',
#    'imagenet',
#    # 'noise',
#    'alexnet',
#    'mobilenet',
#    'vgg',
#    'cifar10',
#    # 'lessnet',
# ]
#
top_dirs = [Path(p) for p in benchmark2ckptdirs['imagenet200']]
run_dirs = natsorted(
    [Path(d) for top_dir in top_dirs for d in top_dir.iterdir() if d.is_dir()], key=str
)
ckpts = natsorted([c for r in run_dirs for c in get_run_ckpts(r)])
# ckpts = [get_run_ckpts(r, filtering=False)[-1] for r in run_dirs]

start = """#!/bin/bash

cleanup() {
  pkill -P $$
  wait
  exit
}

trap cleanup SIGINT SIGTERM\n
"""

if with_methods:
    template = "krenew -- sh -c 'CUDA_VISIBLE_DEVICES={device} python {script} ckpt={ckpt} method={method}' &\n"
    template_cmd = 'python {script} ckpt={ckpt} method={method}\n'
else:
    template = "krenew -- sh -c 'CUDA_VISIBLE_DEVICES={device} python {script} ckpt={ckpt} i={i} n={n}' &\n"
    template_cmd = 'python {script} ckpt={ckpt} i={i} n={n}\n'

delimiter = 'wait $(jobs -p)\n\n'


if with_methods:
    if missing:
        combinations = []
        for c in ckpts:
            h5file = ckpt_to_h5file_path(c)
            methods = check_h5file(h5file)

            try:
                methods.remove('nc_train')
                methods.append('nc')
            except ValueError:
                pass

            try:
                methods.remove('nc_val')
                methods.append('nc')
            except ValueError:
                pass

            combinations.extend([(str(c), m) for m in methods])
    else:
        if method_first:
            combinations = [(str(c), m) for m in methods for c in ckpts]
        else:
            combinations = [(str(c), m) for c in ckpts for m in methods]

    if reverse:
        combinations = combinations[::-1]

    if write_list:
        with open(Path(filename).with_suffix('.txt'), 'w') as f:
            for c, m in combinations:
                f.write(template_cmd.format(script=script, ckpt=c, method=m))
    else:
        with open(filename, 'w') as f:
            f.write(start)

            while combinations:
                for d in devices:
                    try:
                        c, m = combinations.pop()
                    except IndexError:
                        continue

                    f.write(template.format(device=d, script=script, ckpt=c, method=m))

                f.write(delimiter)

else:
    if reverse:
        ckpts = ckpts[::-1]

    with open(filename, 'w') as f:
        f.write(start)

        n = len(ckpts)

        while ckpts:
            for d in devices:
                try:
                    c = ckpts.pop()
                except IndexError:
                    continue
                i = n - len(ckpts)
                f.write(template.format(device=d, script=script, ckpt=c, i=i, n=n))

            f.write(delimiter)

# Make executeable
st = os.stat(filename)
os.chmod(filename, st.st_mode | stat.S_IEXEC)

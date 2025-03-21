import os
import stat
from pathlib import Path

from natsort import natsorted

import utils
from database import key_exists_acc, key_exists_nc, key_exists_ood
from eval_main import get_run_ckpts
from plot_utils import check_h5file

filename = 'run_cifar10.bash'
# script = 'compute_acc_train.py'
script = 'recompute.py'

with_methods = True
method_first = True
reverse = False
missing = False
write_list = True

devices = [0, 1, 2, 3]

accnc_method = ['accnc']
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
methods = accnc_method + odd_methods
methods += ['mnist', 'svhn']

# ckpts = get_previous_ckpts()

# run_dir = Path(
#     '/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_47'
# )
# ckpts = get_run_ckpts(run_dir)
# ckpts = [c for c in ckpts if '200' in str(c) or '500' in str(c)]
# print(ckpts)

top_dir = Path('/home/hauser/neural_collapse/benchmarks/cifar10/ResNet18_32x32/')
#
# top_dir = Path('/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/NCLessNet18/no_noise/1000+_epochs/')
run_dirs = natsorted([d for d in top_dir.iterdir() if d.is_dir()])
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
# top_dirs = [Path(p) for p in benchmark2ckptdirs['cifar10']]
# run_dirs = natsorted(
#     [Path(d) for top_dir in top_dirs for d in top_dir.iterdir() if d.is_dir()], key=str
# )
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
            h5file = utils.ckpt_to_h5file_path(c)
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
        filename = Path(filename).with_suffix('.txt')
        with open(filename, 'w') as f:
            for m in methods:
                for c in ckpts:
                    benchmark = utils.get_benchmark_name(c)
                    model_name = utils.get_model_name(c)
                    run = utils.extract_datetime_from_path(c)
                    epoch = utils.get_epoch_number(c)
                    dataset = benchmark

                    if m == 'accnc':
                        if all(
                            (
                                key_exists_acc(
                                    benchmark, model_name, run, epoch, dataset, 'train'
                                ),
                                key_exists_acc(
                                    benchmark, model_name, run, epoch, dataset, 'val'
                                ),
                                key_exists_nc(
                                    benchmark, model_name, run, epoch, dataset, 'train'
                                ),
                                key_exists_nc(
                                    benchmark, model_name, run, epoch, dataset, 'val'
                                ),
                            )
                        ):
                            continue
                    elif m in ['mnist', 'svhn']:
                        if all(
                            (
                                key_exists_acc(
                                    benchmark, model_name, run, epoch, m, 'val'
                                ),
                                key_exists_nc(
                                    benchmark, model_name, run, epoch, m, 'val'
                                ),
                            )
                        ):
                            continue
                    else:  # OOD
                        if key_exists_ood(benchmark, model_name, run, epoch, m):
                            continue

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

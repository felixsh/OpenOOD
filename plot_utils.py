from collections import defaultdict
import json

import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
from pandas import HDFStore

import path


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

markers = [
    'o',  # Circle
    's',  # Square
    'D',  # Diamond
    '^',  # Upward triangle
    'v',  # Downward triangle
    '<',  # Left triangle
    '>',  # Right triangle
    'p',  # Pentagon
    '*',  # Star
    'h',  # Hexagon
    'X',  # X-shaped marker
    '+',  # Plus sign
    'x',  # X mark
    '|',  # Vertical line
    '_',  # Horizontal line
]

metric_markers = {
    "dice" : 'o',
    "epa" : 's',
    "knn" : 'D',
    "mds" : '^',
    "msp" : 'v',
    "ncscore" : 'p',
    "neco" : '*',
    "nusa" : 'h',
    "odin" : 'X',
    "react" : '+', 
    "vim" : 'x',
}


def load_acc(benchmark_name, run_id, split='val', filter_epochs=None):
    json_dir = path.ckpt_root / benchmark_name / run_id
    with open(json_dir / 'data.json', 'r') as f:
        data = json.load(f)

    acc_values = np.array(data['metrics']['Accuracy'][split]['values'])
    acc_epochs = np.array(data['metrics']['Accuracy'][split]['epochs']) + 1

    if benchmark_name == 'cifar10' and run_id in ['run0', 'run1']:
        acc_epochs -= 1

    if filter_epochs is not None:
        filter_epochs = np.array(filter_epochs)
        return acc_values[np.isin(acc_epochs, filter_epochs)], filter_epochs
    else:
        return acc_values, acc_epochs


def load_noise(benchmark_name, run_id):
    json_dir = path.ckpt_root / benchmark_name / f'run{run_id}'
    with open(json_dir / 'data.json', 'r') as f:
        data = json.load(f)

    return data['metadata']['noise']


def load_acc_nc_ood_mean(benchmark_name,
                   acc_split='val',
                   nc_metric='nc1_cdnv',
                   ood_metric='AUROC'):

    benchmark_dir = path.res_data / benchmark_name
    data = []
    run_ids = []

    for run_dir in benchmark_dir.glob('run*'):
        if benchmark_name == 'imagenet200' and run_dir.name == 'run0':
            continue
        acc_val, acc_epoch = load_acc(benchmark_name, run_dir.name, acc_split)
        run_number = int(run_dir.name[3:])

        if run_number > 7:
            continue

        with HDFStore(run_dir / 'metrics.h5') as store:
            print(str(run_dir))
            nc_df = store.get('/nc')
            nc = nc_df.iloc[0][nc_metric]
            print(nc)

            acc = acc_val[-1]

            ood_keys = list(store.keys())
            ood_keys.remove('/nc')
            near_ood = []
            far_ood = []
            for k in ood_keys:
                ood_df = store.get(k)
                near_ood.append(ood_df.at['nearood', ood_metric])
                print(ood_df.at['nearood', ood_metric])
                far_ood.append(ood_df.at['farood', ood_metric])

            near_ood = np.mean(np.array(near_ood))
            far_ood = np.mean(np.array(far_ood))

            data.append([acc, nc, near_ood, far_ood])
            run_ids.append(run_number)

    return np.array(data), np.array(run_ids)


def load_acc_nc_ood(benchmark_name,
                   acc_split='val',
                   nc_metric='nc1_cdnv',
                   ood_metric='AUROC'):

    benchmark_dir = path.res_data / benchmark_name
    acc_dict = defaultdict(list)
    nc_dict = defaultdict(list)
    nearood_dict = defaultdict(list)
    farood_dict = defaultdict(list)
    run_id_dict = defaultdict(list)

    for run_dir in benchmark_dir.glob('run*'):
        if benchmark_name == 'imagenet200' and run_dir.name == 'run0':
            continue
        ckpt_dirs = natsorted(list(run_dir.glob('e*')), key=str)
        acc_val, acc_epoch = load_acc(benchmark_name, run_dir.name, acc_split)
        run_number = int(run_dir.name[3:])

        for ckpt_dir in ckpt_dirs:
            with HDFStore(ckpt_dir / 'metrics.h5') as store:
                nc_df = store.get('nc')
                nc = nc_df.iloc[0][nc_metric]

                epoch = int(ckpt_dir.name[1:])
                acc = acc_val[acc_epoch == epoch][0]

                ood_keys = list(store.keys())
                ood_keys.remove('/nc')
                for k in ood_keys:
                    ood_df = store.get(k)
                    key = k[1:]

                    acc_dict[key].append(acc)
                    nc_dict[key].append(nc)
                    nearood_dict[key].append(ood_df.at['nearood', ood_metric])
                    farood_dict[key].append(ood_df.at['farood', ood_metric])
                    run_id_dict[key].append(run_number)

    return acc_dict, nc_dict, nearood_dict, farood_dict, run_id_dict

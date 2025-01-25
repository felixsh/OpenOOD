from collections import defaultdict
import json

import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
from pandas import HDFStore

import path
from utils import get_benchmark_name


nc_metrics = (
    'nc1_strong',
    'nc1_weak_between',
    'nc1_weak_within',
    'nc1_cdnv',
    'nc2_equinormness',
    'nc2_equiangularity',
    'gnc2_hyperspherical_uniformity',
    'nc3_self_duality',
    'unc3_uniform_duality',
    'nc4_classifier_agreement',
)

ood_metrics = (
    '/dice',
    '/epa',
    '/knn',
    '/mds',
    '/msp',
    '/ncscore',
    '/neco',
    '/nusa',
    '/odin',
    '/react',
    '/vim',
)

benchmark2loaddirs = {
    'cifar10': (
        '/mrtstorage/users/hauser/openood_res/data/cifar10/ResNet18_32x32/no_noise/300+_epochs',
    ),
    'cifar100': (
        '/mrtstorage/users/hauser/openood_res/data/cifar100/ResNet18_32x32/no_noise/1000+_epochs',
    ),
    'imagenet200': (
        '/mrtstorage/users/hauser/openood_res/data/imagenet200/ResNet18_224x224/no_noise/150+_epochs',
        '/mrtstorage/users/hauser/openood_res/data/imagenet200/ResNet18_224x224/no_noise/200+_epochs',
        '/mrtstorage/users/hauser/openood_res/data/imagenet200/ResNet18_224x224/no_noise/400+_epochs',
        '/mrtstorage/users/hauser/openood_res/data/imagenet200/ResNet18_224x224/no_noise/500+_epochs',
    ),
    'imagenet': (
        '/mrtstorage/users/hauser/openood_res/data/imagenet/ResNet50/no_noise/150+_epochs',
    ),
    'noise': (
        '/mrtstorage/users/hauser/openood_res/data/cifar10/NCResNet18_32x32/noise/300+_epochs',
    ),
    'alexnet': (
        '/mrtstorage/users/hauser/openood_res/data/cifar10/NCAlexNet/no_noise/300+_epochs',
    ),
    'mobilenet': (
        '/mrtstorage/users/hauser/openood_res/data/cifar10/NCMobileNetV2/no_noise/300+_epochs',
    ),
    'vgg': (
        '/mrtstorage/users/hauser/openood_res/data/cifar10/NCVGG16/no_noise/300+_epochs',
    ),
    'lessnet': (
        '/mrtstorage/users/hauser/openood_res/data/cifar100/NCLessNet18/no_noise/1000+_epochs',
    ),
}


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


def numpify_dict(dict_of_lists):
    dict_of_arrays = {key: np.array(value) for key, value in dict_of_lists.items()}
    return dict_of_arrays


def load_acc(run_data_dir, filter_epochs=None):
    """Return accuracy values with corresponding epochs for run from data.json."""

    run_ckpt_dir = path.ckpt_root / run_data_dir.relative_to(path.res_data)
    with open(run_ckpt_dir / 'data.json', 'r') as f:
        data = json.load(f)

    acc = {}
    for split, acc_dict in data['metrics']['Accuracy'].items():
        acc[split] = numpify_dict(acc_dict)
        acc[split]['epochs'] += 1

        if filter_epochs is not None:
            filter_epochs = np.array(filter_epochs)
            idx = np.isin(acc[split]['epochs'], filter_epochs)
            acc[split]['epochs'] = acc[split]['epochs'][idx]
            acc[split]['values'] = acc[split]['values'][idx]

    # if benchmark_name == 'cifar10' and run_id in ['run0', 'run1']:
    #     acc_epochs -= 1

    return acc


def load_nc(run_data_dir, nc_split='val', benchmark=None):
    """Return nc metrics with corresponding epochs for run from hdf5 files."""

    if nc_split=='train':
        nc_key = '/nc_train'
    elif nc_split=='val':
        nc_key = '/nc_val'
    else:
        raise NotImplementedError

    if benchmark == 'imagenet':
        nc_key = '/nc'

    nc = defaultdict(list)
    epochs = []

    for h5file in natsorted(list(run_data_dir.glob('e*.h5')), key=str):
        epoch = int(h5file.stem[1:])
        epochs.append(epoch)

        with HDFStore(h5file, mode='r') as store:
            df = store.get(nc_key)
            for metric, value in df.items():
                nc[metric].append(value)
            
    return numpify_dict(nc), np.array(epochs)


def load_ood(run_data_dir, ood_metric='AUROC'):
    """Return ood metrics with corresponding epochs for run from hdf5 files."""
    nearood = defaultdict(list)
    farood = defaultdict(list)
    epochs = []

    for h5file in natsorted(list(run_data_dir.glob('e*.h5')), key=str):
        epoch = int(h5file.stem[1:])
        epochs.append(epoch)

        with HDFStore(h5file, mode='r') as store:
            ood_keys = list(store.keys())
            try:
                ood_keys.remove('/nc')
            except ValueError:
                pass
            ood_keys.remove('/nc_train')
            ood_keys.remove('/nc_val')
            for k in ood_keys:
                df = store.get(k)
                key = k[1:]
                nearood[key].append(df.at['nearood', ood_metric])
                farood[key].append(df.at['farood', ood_metric])
        
    return numpify_dict(nearood), numpify_dict(farood), np.array(epochs)


def load_nc_ood(run_data_dir, nc_split='val', ood_metric='AUROC'):
    """Return ood metrics with corresponding epochs for run from hdf5 files."""
    nearood = defaultdict(list)
    farood = defaultdict(list)
    epochs = []

    if nc_split=='train':
        nc_key = '/nc_train'
    elif nc_split=='val':
        nc_key = '/nc_val'
    else:
        raise NotImplementedError

    nc = defaultdict(list)

    for h5file in natsorted(list(run_data_dir.glob('e*.h5')), key=str):
        epoch = int(h5file.stem[1:])
        epochs.append(epoch)

        with HDFStore(h5file, mode='r') as store:

            nc_df = store.get(nc_key)
            for metric, value in nc_df.items():
                nc[metric].append(value)

            ood_keys = list(store.keys())
            try:
                ood_keys.remove('/nc')
            except ValueError:
                pass

            try:
                ood_keys.remove('/nc_train')
                ood_keys.remove('/nc_val')
            except ValueError:
                pass

            # Check if all keys are present
            assert set(ood_metrics) <= set(ood_keys), f'Missing keys {set(ood_metrics) - set(ood_keys)} in file {h5file}'

            for k in ood_keys:
                df = store.get(k)
                key = k[1:]
                nearood[key].append(df.at['nearood', ood_metric])
                farood[key].append(df.at['farood', ood_metric])

    return nc, numpify_dict(nearood), numpify_dict(farood), epochs


def check_run_data(run_data_dir):
    """Check data of run for completeness."""
    for h5file in natsorted(list(run_data_dir.glob('e*.h5')), key=str):
        benchmark = get_benchmark_name(h5file)
        with HDFStore(h5file, mode='r') as store:
            keys = list(store.keys())

            if not '/nc_train' in keys:
                print(f'Missing /nc_train in benchmark {benchmark} in file {h5file}')
            if not '/nc_val' in keys:
                print(f'Missing /nc_val in benchmark {benchmark} in file {h5file}')

            try:
                keys.remove('/nc')
            except ValueError:
                pass

            try:
                keys.remove('/nc_train')
                keys.remove('/nc_val')
            except ValueError:
                pass

            # Check if all keys are present
            if not set(ood_metrics) <= set(keys):
                print(f'Missing keys {set(ood_metrics) - set(keys)} in file {h5file}')


def load_noise(run_data_dir):
    run_ckpt_dir = path.ckpt_root / run_data_dir.relative_to(path.res_data)
    with open(run_ckpt_dir / 'data.json', 'r') as f:
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

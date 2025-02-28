import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted
from pandas import HDFStore

import path

nc_metrics_cov = (
    # 'nc1_strong',
    'nc1_weak_between',
    'nc1_weak_within',
    'nc1_cdnv_cov',
    'nc2_equinormness_cov',
    'nc2_equiangularity_cov',
    'gnc2_hyperspherical_uniformity_cov',
    'nc3_self_duality',
    'unc3_uniform_duality_cov',
    'nc4_classifier_agreement',
)

nc_metrics_mean = (
    # 'nc1_strong',
    'nc1_weak_between',
    'nc1_weak_within',
    'nc1_cdnv_mean',
    'nc2_equinormness_mean',
    'nc2_equiangularity_mean',
    'gnc2_hyperspherical_uniformity_mean',
    'nc3_self_duality',
    'unc3_uniform_duality_mean',
    'nc4_classifier_agreement',
)

ood_methods = (
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


benchmark2ckptdirs = {
    'cifar10': (
        '/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs',
    ),
    'cifar100': (
        '/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs',
    ),
    'imagenet200': (
        '/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs',
        '/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs',
        '/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs',
        '/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs',
    ),
    'imagenet': (
        '/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet/ResNet50/no_noise/150+_epochs',
    ),
    'noise': (
        '/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs',
    ),
    'alexnet': (
        '/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs',
    ),
    'mobilenet': (
        '/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs',
    ),
    'vgg': (
        '/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs',
    ),
    'lessnet': (
        '/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/NCLessNet18/no_noise/1000+_epochs',
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
    'dice': 'o',
    'epa': 's',
    'knn': 'D',
    'mds': '^',
    'msp': 'v',
    'ncscore': 'p',
    'neco': '*',
    'nusa': 'h',
    'odin': 'X',
    'react': '+',
    'vim': 'x',
}


def numpify_dict(dict_of_lists):
    dict_of_arrays = {key: np.array(value) for key, value in dict_of_lists.items()}
    return dict_of_arrays


def mean_ood_1dict(d):
    return np.mean([v for v in d.values()], axis=0)


def mean_ood_2dict(nearood_dict, farood_dict):
    nood_values = mean_ood_1dict(nearood_dict)
    food_values = mean_ood_1dict(farood_dict)
    x = np.vstack((nood_values, food_values))
    return x.mean(axis=0)


def load_acc(run_data_dir, filter_epochs=None, benchmark=None):
    """Return accuracy values with corresponding epochs for run from data.json."""

    run_ckpt_dir = path.ckpt_root / run_data_dir.relative_to(path.res_data)
    with open(run_ckpt_dir / 'data.json', 'r') as f:
        data = json.load(f)

    acc = {}
    for split, acc_dict in data['metrics']['Accuracy'].items():
        acc[split] = numpify_dict(acc_dict)

        # if benchmark in ('imagenet', 'noise', 'cifar10', 'imagenet200'):
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

    if nc_split == 'train':
        nc_key = '/nc_train'
    elif nc_split == 'val':
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


def load_nc_ood(run_data_dir, nc_split='val', ood_metric='AUROC', benchmark=None):
    """Return ood metrics with corresponding epochs for run from hdf5 files."""
    nearood = defaultdict(list)
    farood = defaultdict(list)
    epochs = []

    if nc_split == 'train':
        nc_key = '/nc_train'
    elif nc_split == 'val':
        nc_key = '/nc_val'
    else:
        raise NotImplementedError

    nc = defaultdict(list)

    if benchmark == 'cifar100':
        h5file_list = natsorted(list(run_data_dir.glob('e*.h5')), key=str)[:-1]
    else:
        h5file_list = natsorted(list(run_data_dir.glob('e*.h5')), key=str)

    if benchmark == 'noise':
        h5file_list = [h5file_list[-1]]

    for h5file in h5file_list:
        epoch = int(h5file.stem[1:])
        epochs.append(epoch)

        with HDFStore(h5file, mode='r') as store:
            nc_df = store.get(nc_key)
            for metric, value in nc_df.items():
                if metric != 'nc1_strong':
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

            try:
                ood_keys.remove('/acc')
            except ValueError:
                pass

            # Check if all keys are present
            assert set(ood_methods) <= set(ood_keys), (
                f'Missing keys {set(ood_methods) - set(ood_keys)} in file {h5file}'
            )

            for k in ood_keys:
                df = store.get(k)
                key = k[1:]
                nearood[key].append(df.at['nearood', ood_metric])
                farood[key].append(df.at['farood', ood_metric])

    return nc, numpify_dict(nearood), numpify_dict(farood), epochs


def check_run_data(run_data_dir):
    """Check data of run for completeness."""
    for h5file in natsorted(list(run_data_dir.glob('e*.h5')), key=str):
        with HDFStore(h5file, mode='r') as store:
            keys = list(store.keys())

            if '/nc_train' not in keys:
                print(f'Missing /nc_train in file {h5file}')
            if '/nc_val' not in keys:
                print(f'Missing /nc_val in file {h5file}')

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
            if not set(ood_methods) <= set(keys):
                print(f'Missing keys {set(ood_methods) - set(keys)} in file {h5file}')


def load_acc_train(run_data_dir, benchmark=None):
    """Return ood metrics with corresponding epochs for run from hdf5 files."""

    if benchmark == 'cifar100':
        h5file_list = natsorted(list(run_data_dir.glob('e*.h5')), key=str)[:-1]
    else:
        h5file_list = natsorted(list(run_data_dir.glob('e*.h5')), key=str)

    acc_train = []

    for h5file in h5file_list:
        with HDFStore(h5file, mode='r') as store:
            df = store.get('/acc')
            acc_train.append(df.at['id', 'train'])

    return np.array(acc_train)


def load_noise(run_data_dir):
    run_ckpt_dir = path.ckpt_root / run_data_dir.relative_to(path.res_data)
    with open(run_ckpt_dir / 'data.json', 'r') as f:
        data = json.load(f)

    return data['metadata']['noise']


def load_acc_nc_ood_mean(
    benchmark_name, acc_split='val', nc_metric='nc1_cdnv', ood_metric='AUROC'
):
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


def load_acc_nc_ood(
    benchmark_name, acc_split='val', nc_metric='nc1_cdnv', ood_metric='AUROC'
):
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


class IncompleteError(Exception):
    pass


def load_benchmark_data(
    benchmark_name,
    nc_split='val',
    ood_metric='AUROC',
):
    # Get run dirs
    main_dirs = benchmark2loaddirs[benchmark_name]
    main_dirs = [Path(p) for p in main_dirs]
    run_dirs = natsorted(
        [
            subdir
            for p in main_dirs
            if p.is_dir()
            for subdir in p.iterdir()
            if subdir.is_dir()
        ],
        key=str,
    )
    print('number of runs', len(run_dirs))

    save_dir = path.res_plots / main_dirs[0].relative_to(path.res_data).parents[-2]
    save_dir.mkdir(parents=True, exist_ok=True)

    # Collect data
    epochs = []
    run_ids = []
    acc_val = []
    acc_train = []
    nc = defaultdict(list)
    nood = defaultdict(list)
    food = defaultdict(list)

    # for run_dir in run_dirs:
    #     check_run_data(run_dir)

    for run_id, run_dir in enumerate(run_dirs):
        nc_dict, nearood_dict, farood_dict, epochs_ = load_nc_ood(
            run_dir, nc_split=nc_split, ood_metric=ood_metric, benchmark=benchmark_name
        )
        acc_val_ = load_acc(run_dir, filter_epochs=epochs_, benchmark=benchmark_name)
        acc_val_ = list(acc_val_['val']['values'])

        epochs.extend(epochs_)
        run_ids.extend([run_id for _ in range(len(epochs_))])
        acc_val.extend(acc_val_)

        for k, v in nc_dict.items():
            nc[k].extend(v)

        for k, v in nearood_dict.items():
            nood[k].extend(v)

        for k, v in farood_dict.items():
            food[k].extend(v)

        acc_train_ = load_acc_train(run_dir, benchmark=benchmark_name)
        acc_train.extend(acc_train_)

    run_ids = np.array(run_ids)
    epochs = np.array(epochs)
    acc_val = np.array(acc_val)
    acc_train = np.array(acc_train)
    nc = numpify_dict(nc)
    nood = numpify_dict(nood)
    food = numpify_dict(food)

    return run_ids, epochs, acc_val, acc_train, nc, nood, food, save_dir


def load_noise_data(
    nc_split='val',
    ood_metric='AUROC',
):
    # Get run dirs
    main_dirs = benchmark2loaddirs['noise']
    main_dirs = [Path(p) for p in main_dirs]
    run_dirs = natsorted(
        [
            subdir
            for p in main_dirs
            if p.is_dir()
            for subdir in p.iterdir()
            if subdir.is_dir()
        ],
        key=str,
    )
    print('number of runs', len(run_dirs))

    save_dir = path.res_plots / main_dirs[0].relative_to(path.res_data).parents[-2]
    save_dir.mkdir(parents=True, exist_ok=True)

    # Collect data
    noise_lvl = np.array([load_noise(r) for r in run_dirs])
    epochs = []
    acc_val = []
    acc_train = []
    nc = defaultdict(list)
    nood = defaultdict(list)
    food = defaultdict(list)

    # for run_dir in run_dirs:
    #     check_run_data(run_dir)

    for run_dir in run_dirs:
        nc_dict, nearood_dict, farood_dict, epochs_ = load_nc_ood(
            run_dir, nc_split=nc_split, ood_metric=ood_metric, benchmark='noise'
        )
        acc_ = load_acc(run_dir, filter_epochs=epochs_, benchmark='noise')
        acc_ = list(acc_['val']['values'])

        epochs.append(epochs_[-1])
        acc_val.append(acc_[-1])

        for k, v in nc_dict.items():
            nc[k].append(v[-1])

        for k, v in nearood_dict.items():
            nood[k].append(v[-1])

        for k, v in farood_dict.items():
            food[k].append(v[-1])

        acc_train_ = load_acc_train(run_dir, benchmark='noise')
        acc_train.append(acc_train_[-1])

    epochs = np.array(epochs)
    acc_val = np.array(acc_val)
    acc_train = np.array(acc_train)
    nc = numpify_dict(nc)
    nood = numpify_dict(nood)
    food = numpify_dict(food)

    return noise_lvl, epochs, acc_val, acc_train, nc, nood, food, save_dir

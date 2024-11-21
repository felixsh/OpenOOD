from collections import defaultdict

import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
from scipy.stats import spearmanr
import seaborn as sns  
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from summarize_json import summarize_json

import path
from plot_utils import colors, metric_markers, markers
from plot_utils import load_acc_nc_ood, load_acc_nc_ood_mean, load_acc, load_noise


def plot_acc_nc_ood(benchmark_name,
                    acc_split='val',
                    nc_metric='nc1_cdnv',
                    ood_metric='AUROC'):

    data_mean, run_ids_mean = load_acc_nc_ood_mean(benchmark_name,
                                        acc_split=acc_split,
                                        nc_metric=nc_metric,
                                        ood_metric=ood_metric)

    acc, nc, nearood, farood, run_ids = load_acc_nc_ood(benchmark_name,
                                        acc_split=acc_split,
                                        nc_metric=nc_metric,
                                        ood_metric=ood_metric)

    labels = ['acc', 'nc', 'nearood', 'farood']

    def plot_cut(x, y, z, resolution=100, cuts=3):
        kde = KDEMultivariate(data_mean[:, [x, y, z]], 'ccc')

        x_lin = np.linspace(data_mean[:, x].min(), data_mean[:, x].max(), resolution)
        y_lin = np.linspace(data_mean[:, y].min(), data_mean[:, y].max(), resolution)
        z_lin = np.linspace(data_mean[:, z].min(), data_mean[:, z].max(), cuts**2)

        X, Y = np.meshgrid(x_lin, y_lin)
        ones = np.ones_like(X)

        res = []
        for z_ in z_lin:
            Z = z_ * ones
            points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
            pdf = kde.pdf(points).reshape(X.shape)
            res.append(pdf)

        fig, axes = plt.subplots(cuts, cuts, figsize=(10, 10))

        for ax, pdf in zip(axes.ravel(), res):
            ax.contourf(X, Y, pdf)
            ax.axis('off')

        plt.tight_layout()
        fig.suptitle(f'{labels[z]}  {labels[x]}-{labels[y]}')
        
        save_path = path.res_plots / benchmark_name
        save_path.mkdir(exist_ok=True, parents=True)
        filename = f'acc_{acc_split}_{nc_metric}_{ood_metric}_{labels[z]}.png'
        plt.savefig(save_path / filename, bbox_inches='tight')
        plt.close()

    plot_cut(0, 2, 1)
    plot_cut(0, 3, 1)

    def filter_and_join_dict(dict_data, keys_to_exclude):
        # Filter out the specified keys and get the remaining values
        filtered_values = [value for key, value in dict_data.items() if key not in keys_to_exclude]
        
        # Flatten and join the remaining values into a single numpy array
        joined_array = np.concatenate([np.array(v) for v in filtered_values])
        
        return joined_array

    def plot_cut_all(acc, nc, ood, ood_label, resolution=100, cuts=3):
        filt_keys_out = ['neco', 'mds', 'nusa']
        acc = filter_and_join_dict(acc, filt_keys_out)
        ood = filter_and_join_dict(ood, filt_keys_out)
        nc = filter_and_join_dict(nc, filt_keys_out)

        inp = np.stack([acc, ood, nc], axis=0).T
        kde = KDEMultivariate(inp, 'ccc')

        x_lin = np.linspace(acc.min(), acc.max(), resolution)
        y_lin = np.linspace(ood.min(), ood.max(), resolution)
        z_lin = np.linspace(nc.min(), nc.max(), cuts**2)

        X, Y = np.meshgrid(x_lin, y_lin)
        ones = np.ones_like(X)

        res = []
        for z_ in z_lin:
            Z = z_ * ones
            points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
            pdf = kde.pdf(points).reshape(X.shape)
            res.append(pdf)

        fig, axes = plt.subplots(cuts, cuts, figsize=(10, 10))

        for ax, pdf, z_ in zip(axes.ravel(), res, z_lin):
            ax.contourf(X, Y, pdf)
            ax.axis('off')
            ax.text(0.95, 0.05, f'{z_:.2f}', ha='right', va='top', transform=ax.transAxes)

        plt.tight_layout()
        fig.suptitle(f'P(acc, ood | nc)')
        
        save_path = path.res_plots / benchmark_name
        save_path.mkdir(exist_ok=True, parents=True)
        filename = f'all_acc_{acc_split}_{nc_metric}_{ood_metric}_{ood_label}.png'
        plt.savefig(save_path / filename, bbox_inches='tight')
        plt.close()
    
    plot_cut_all(acc, nc, nearood, 'nearood')
    plot_cut_all(acc, nc, farood, 'farood')

    def plot_corr_mean(z, label):
        acc = data_mean[:, 0]
        nc = data_mean[:, 1]
        ood = data_mean[:, z]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        c = [colors[i] for i in run_ids_mean]

        axes.ravel()[0].scatter(acc, nc, c=c, marker='o')
        axes.ravel()[0].set_xlabel(f'acc {acc_split}')
        axes.ravel()[0].set_ylabel(nc_metric)

        axes.ravel()[1].scatter(acc, ood, c=c, marker='o')
        axes.ravel()[1].set_xlabel(f'acc {acc_split}')
        axes.ravel()[1].set_ylabel(f'{ood_metric} {label}')

        axes.ravel()[2].scatter(nc, ood, c=c, marker='o')
        axes.ravel()[2].set_xlabel(nc_metric)
        axes.ravel()[2].set_ylabel(f'{ood_metric} {label}')

        plt.tight_layout()

        save_path = path.res_plots / benchmark_name
        save_path.mkdir(exist_ok=True, parents=True)
        filename = f'corr_{label}.png'
        plt.savefig(save_path / filename, bbox_inches='tight')
        plt.close()

    plot_corr_mean(2, 'nearood')
    plot_corr_mean(3, 'farood')

    def plot_corr_method(acc, nc, ood, ood_label, run_ids):
        for k in acc.keys():
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f'{k} {ood_label}')

            c = [colors[i] for i in run_ids[k]]

            axes.ravel()[0].scatter(acc[k], nc[k], c=c, marker='o')
            axes.ravel()[0].set_xlabel(f'acc {acc_split}')
            axes.ravel()[0].set_ylabel(nc_metric)

            axes.ravel()[1].scatter(acc[k], ood[k], c=c, marker='o')
            axes.ravel()[1].set_xlabel(f'acc {acc_split}')
            axes.ravel()[1].set_ylabel(f'{ood_metric}')

            axes.ravel()[2].scatter(nc[k], ood[k], c=c, marker='o')
            axes.ravel()[2].set_xlabel(nc_metric)
            axes.ravel()[2].set_ylabel(f'{ood_metric}')

            plt.tight_layout()

            save_path = path.res_plots / benchmark_name / f'corr_{ood_label}'
            save_path.mkdir(exist_ok=True, parents=True)
            filename = f'{k}.png'
            plt.savefig(save_path / filename, bbox_inches='tight')
            plt.close()
    
    plot_corr_method(acc, nc, nearood, 'nearood', run_ids)
    plot_corr_method(acc, nc, farood, 'farood', run_ids)

    def plot_corr_matrix():
        df_near = pd.DataFrame(data_mean[:, [0, 1, 2]])
        df_far = pd.DataFrame(data_mean[:, [0, 1, 3]])

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        im0 = axes[0].matshow(df_near.corr())
        axes[0].set_title('nearood')
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].matshow(df_far.corr())
        axes[1].set_title('farood')
        fig.colorbar(im1, ax=axes[1])

        plt.tight_layout()

        save_path = path.res_plots / benchmark_name
        save_path.mkdir(exist_ok=True, parents=True)
        filename = f'corr_matrix.png'
        plt.savefig(save_path / filename, bbox_inches='tight')
        plt.close()

    plot_corr_matrix()


def plot_ood(benchmark_name,
             run_id,
             ood_metric='AUROC'):
    
    nc_ood_methods = ['nusa', 'vim', 'ncscore', 'neco', 'epa']

    main_dir = path.res_data / benchmark_name / run_id
    ckpt_dirs = natsorted(list(main_dir.glob('e*')), key=str)

    for ckpt_dir in ckpt_dirs:
        with pd.HDFStore(ckpt_dir / 'metrics.h5') as store:
            ood_keys = list(store.keys())
            ood_keys.remove('/nc')
            for k in ood_keys:
                color = colors[1] if k[1:] in nc_ood_methods else colors[0]
                label = 'nc method' if k[1:] in nc_ood_methods else 'baseline method'

                ood_df = store.get(k)
                near_ood = ood_df.at['nearood', ood_metric]
                far_ood = ood_df.at['farood', ood_metric]
                plt.plot(near_ood, far_ood, 'o', color=color, label=label)
        
        ax = plt.gca()
        x0 = ax.get_xlim()[0]
        y0 = ax.get_ylim()[0]
        ax.axline((x0, y0), slope=1, ls='--', color='k', alpha=0.5, zorder=1)

        plt.title(f'{benchmark_name} {run_id} {ckpt_dir.name} {ood_metric}')
        plt.xlabel('nearood')
        plt.ylabel('farood')
        
        # https://stackoverflow.com/a/13589144
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        ax.set_aspect('equal')

        save_path = path.res_plots / benchmark_name / run_id
        save_path.mkdir(exist_ok=True, parents=True)
        filename = f'{ood_metric}_{ckpt_dir.name}.png'
        plt.savefig(save_path / filename, bbox_inches='tight')
        plt.close()



def plot_ood_combined(benchmark_name,
                      run_id,
                      ood_metric='AUROC'):
    
    nc_ood_methods = ['nusa', 'vim', 'ncscore', 'neco', 'epa']

    main_dir = path.res_data / benchmark_name / run_id
    ckpt_dirs = natsorted(list(main_dir.glob('e*')), key=str)

    epoch = []
    near_ood = defaultdict(list)
    far_ood = defaultdict(list)

    for ckpt_dir in ckpt_dirs:
        with pd.HDFStore(ckpt_dir / 'metrics.h5') as store:
            epoch.append(int(ckpt_dir.name[1:]))
            ood_keys = list(store.keys())
            ood_keys.remove('/nc')
            for k in ood_keys:
                ood_df = store.get(k)
                key = k[1:]
                near_ood[key].append(ood_df.at['nearood', ood_metric])
                far_ood[key].append(ood_df.at['farood', ood_metric])

    for k in near_ood.keys():
        color = colors[1] if k in nc_ood_methods else colors[0]
        label = 'nc method' if k in nc_ood_methods else 'baseline method'
        plt.plot(near_ood[k], far_ood[k], '-', color=color, alpha=0.3)
        plt.plot(near_ood[k], far_ood[k], 'o', color=color, label=label)

    ax = plt.gca()
    x0 = ax.get_xlim()[0]
    y0 = ax.get_ylim()[0]
    ax.axline((x0, y0), slope=1, ls='--', color='k', alpha=0.5, zorder=1)

    plt.title(f'{benchmark_name} {run_id} {ckpt_dir.name} {ood_metric}')
    plt.xlabel('nearood')
    plt.ylabel('farood')
        
    # https://stackoverflow.com/a/13589144
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    ax.set_aspect('equal')

    save_path = path.res_plots / benchmark_name / run_id
    save_path.mkdir(exist_ok=True, parents=True)
    filename = f'{ood_metric}_near_far.png'
    plt.savefig(save_path / filename, bbox_inches='tight')
    plt.close()


def plot_acc_ood_avg(benchmark_name,
                     acc_split='val',  # 'train' or 'val'
                     ood_metric='AUROC',
                     far=False,
                     x_axis = "epoch" 
                     ):
    main_dir = path.res_data / benchmark_name
    run_dirs = natsorted(list(main_dir.glob('run*')), key=str)

    # Initialize data structures
    acc_dict = defaultdict(list)
    near_ood_dict = defaultdict(lambda: defaultdict(list))  # ood_key -> epoch -> list of values
    far_ood_dict = defaultdict(lambda: defaultdict(list))

    for run_dir in run_dirs:
        print(run_dir.name)
        if benchmark_name == 'cifar10' and run_dir.name in ['run0', 'run1']:
            continue
        if benchmark_name == 'imagenet200' and run_dir.name in ['run0',]:
            continue
        # For each run, get the ckpt_dirs
        ckpt_dirs = natsorted(list(run_dir.glob('e*')), key=str)
        ckpt_dirs = [p for p in ckpt_dirs if 'e9' not in str(p)]
        epochs = []
        for ckpt_dir in ckpt_dirs:
            epoch_num = int(ckpt_dir.name[1:])
            epochs.append(epoch_num)
            with pd.HDFStore(ckpt_dir / 'metrics.h5') as store:
                ood_keys = list(store.keys())
                ood_keys.remove('/nc')
                for k in ood_keys:
                    ood_df = store.get(k)
                    near_ood_dict[k][epoch_num].append(ood_df.at['nearood', ood_metric])
                    far_ood_dict[k][epoch_num].append(ood_df.at['farood', ood_metric])
        # Get accuracy for this run
        acc_values, _ = load_acc(benchmark_name, run_dir.name, split=acc_split, filter_epochs=epochs)
        for epoch_num, acc_value in zip(epochs, acc_values):
            acc_dict[epoch_num].append(acc_value)

    # Get all unique epochs and sort them
    # epochs = sorted(acc_dict.keys())
    # log_epochs = np.log(epochs)
    # log_epochs = np.where(np.isinf(log_epochs), 0, log_epochs)
    # print(log_epochs)

    # Compute average accuracy per epoch
    avg_acc = [np.mean(acc_dict[epoch]) for epoch in epochs]

    # Get all OOD keys
    ood_keys = near_ood_dict.keys()

    # Compute average near OOD and far OOD metrics per epoch
    avg_near_ood = {
        k: [np.mean(near_ood_dict[k][epoch]) if epoch in near_ood_dict[k] else np.nan for epoch in epochs]
        for k in ood_keys
    }
    avg_far_ood = {
        k: [np.mean(far_ood_dict[k][epoch]) if epoch in far_ood_dict[k] else np.nan for epoch in epochs]
        for k in ood_keys
    }

    # epochs = log_epochs 

    # Plotting
    # plt.title(f'{benchmark_name} {"far" if far else "near"}')
    for ood_key in ood_keys:
        print(ood_key)
        y_values = avg_far_ood[ood_key] if far else avg_near_ood[ood_key]

        zipped =  zip(epochs, y_values) if x_axis == "epoch" else zip(avg_acc, y_values)

        # Remove NaNs from data
        valid_data = [(x, y) for x, y in zipped if not np.isnan(y)]
        if not valid_data:
            continue
        x_values, y_values = zip(*valid_data)
        plt.plot(x_values, y_values, '-', alpha=0.3, color=colors[1] if far else colors[0])
        plt.plot(x_values, y_values, metric_markers[ood_key[1:]],
                 color=colors[1] if far else colors[0], label=ood_key[1:])

    if x_axis == "epoch":
        plt.xlabel('epoch')
        plt.gca().set_xscale('log')
    else:
        plt.xlabel(f'acc {acc_split}')
    plt.ylabel(ood_metric)


    # Create legend without duplicates
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc=2)

    # Save the plot
    save_path = path.res_plots / benchmark_name
    save_path.mkdir(exist_ok=True, parents=True)
    filename = f'acc_{acc_split}_{ood_metric}_{"far" if far else "near"}_{"avg"}_{"epoch" if x_axis == "epoch" else "acc"}.png'
    plt.savefig(save_path / filename, bbox_inches='tight')
    plt.close()


def plot_run_specific(benchmark_name, run_id):
    plot_nc_ood(benchmark_name, run_id)
    plot_acc_ood(benchmark_name, run_id, acc_split='val')
    # plot_acc_ood(benchmark_name, run_id, acc_split='train')
    # plot_nc(benchmark_name, run_id)
    plot_ood(benchmark_name, run_id)
    plot_ood_combined(benchmark_name, run_id)


def plot_correlation(benchmark_name,
                     acc_split='val',
                     nc_metric='nc1_cdnv',
                     ood_metric='AUROC'):

    data_mean, run_ids_mean = load_acc_nc_ood_mean(benchmark_name,
                                        acc_split=acc_split,
                                        nc_metric=nc_metric,
                                        ood_metric=ood_metric)

    acc, nc, nearood, farood, run_ids = load_acc_nc_ood(benchmark_name,
                                        acc_split=acc_split,
                                        nc_metric=nc_metric,
                                        ood_metric=ood_metric)

    labels = ['acc', 'nc', 'nearood', 'farood']

    def plot_cut(x, y, z, resolution=100, cuts=3):
        kde = KDEMultivariate(data_mean[:, [x, y, z]], 'ccc')

        x_lin = np.linspace(data_mean[:, x].min(), data_mean[:, x].max(), resolution)
        y_lin = np.linspace(data_mean[:, y].min(), data_mean[:, y].max(), resolution)
        z_lin = np.linspace(data_mean[:, z].min(), data_mean[:, z].max(), cuts**2)

        X, Y = np.meshgrid(x_lin, y_lin)
        ones = np.ones_like(X)

        res = []
        for z_ in z_lin:
            Z = z_ * ones
            points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
            pdf = kde.pdf(points).reshape(X.shape)
            res.append(pdf)

        fig, axes = plt.subplots(cuts, cuts, figsize=(10, 10))

        for ax, pdf in zip(axes.ravel(), res):
            ax.contourf(X, Y, pdf)
            ax.axis('off')

        plt.tight_layout()
        fig.suptitle(f'{labels[z]}  {labels[x]}-{labels[y]}')
        
        save_path = path.res_plots / benchmark_name
        save_path.mkdir(exist_ok=True, parents=True)
        filename = f'acc_{acc_split}_{nc_metric}_{ood_metric}_{labels[z]}.png'
        plt.savefig(save_path / filename, bbox_inches='tight')
        plt.show()

    plot_cut(0, 2, 1)
    plot_cut(0, 3, 1)

    def filter_and_join_dict(dict_data, keys_to_exclude):
        # Filter out the specified keys and get the remaining values
        filtered_values = [value for key, value in dict_data.items() if key not in keys_to_exclude]
        
        # Flatten and join the remaining values into a single numpy array
        joined_array = np.concatenate([np.array(v) for v in filtered_values])
        
        return joined_array

    def plot_cut_all(acc, nc, ood, ood_label, resolution=100, cuts=3):
        filt_keys_out = ['neco', 'mds', 'nusa']
        acc = filter_and_join_dict(acc, filt_keys_out)
        ood = filter_and_join_dict(ood, filt_keys_out)
        nc = filter_and_join_dict(nc, filt_keys_out)

        inp = np.stack([acc, ood, nc], axis=0).T
        kde = KDEMultivariate(inp, 'ccc')

        x_lin = np.linspace(acc.min(), acc.max(), resolution)
        y_lin = np.linspace(ood.min(), ood.max(), resolution)
        z_lin = np.linspace(nc.min(), nc.max(), cuts**2)

        X, Y = np.meshgrid(x_lin, y_lin)
        ones = np.ones_like(X)

        res = []
        for z_ in z_lin:
            Z = z_ * ones
            points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
            pdf = kde.pdf(points).reshape(X.shape)
            res.append(pdf)

        fig, axes = plt.subplots(cuts, cuts, figsize=(10, 10))

        for ax, pdf, z_ in zip(axes.ravel(), res, z_lin):
            ax.contourf(X, Y, pdf)
            ax.axis('off')
            ax.text(0.95, 0.05, f'{z_:.2f}', ha='right', va='top', transform=ax.transAxes)

        plt.tight_layout()
        fig.suptitle(f'P(acc, ood | nc)')
        
        save_path = path.res_plots / benchmark_name
        save_path.mkdir(exist_ok=True, parents=True)
        filename = f'all_acc_{acc_split}_{nc_metric}_{ood_metric}_{ood_label}.png'
        plt.savefig(save_path / filename, bbox_inches='tight')
        plt.close()
    
    plot_cut_all(acc, nc, nearood, 'nearood')
    plot_cut_all(acc, nc, farood, 'farood')

    def plot_corr_mean(z, label):
        acc = data_mean[:, 0]
        nc = data_mean[:, 1]
        ood = data_mean[:, z]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        c = [colors[i] for i in run_ids_mean]

        axes.ravel()[0].scatter(acc, nc, c=c, marker='o')
        axes.ravel()[0].set_xlabel(f'acc {acc_split}')
        axes.ravel()[0].set_ylabel(nc_metric)

        axes.ravel()[1].scatter(acc, ood, c=c, marker='o')
        axes.ravel()[1].set_xlabel(f'acc {acc_split}')
        axes.ravel()[1].set_ylabel(f'{ood_metric} {label}')

        axes.ravel()[2].scatter(nc, ood, c=c, marker='o')
        axes.ravel()[2].set_xlabel(nc_metric)
        axes.ravel()[2].set_ylabel(f'{ood_metric} {label}')

        plt.tight_layout()

        save_path = path.res_plots / benchmark_name
        save_path.mkdir(exist_ok=True, parents=True)
        filename = f'corr_{label}.png'
        plt.savefig(save_path / filename, bbox_inches='tight')
        plt.close()

    plot_corr_mean(2, 'nearood')
    plot_corr_mean(3, 'farood')

    def plot_corr_mean(z, label):
        acc_data = data_mean[:, 0]
        nc_data = data_mean[:, 1]
        ood_data = data_mean[:, z]
        
        # Compute Spearman correlations
        corr_acc_nc, p_acc_nc = spearmanr(acc_data, nc_data)
        corr_acc_ood, p_acc_ood = spearmanr(acc_data, ood_data)
        corr_nc_ood, p_nc_ood = spearmanr(nc_data, ood_data)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        c = [colors[i] for i in run_ids_mean]
        
        axes[0].scatter(acc_data, nc_data, c=c, marker='o')
        axes[0].set_xlabel(f'acc {acc_split}')
        axes[0].set_ylabel(nc_metric)
        axes[0].set_title(f'Spearman r={corr_acc_nc:.2f}, p={p_acc_nc:.2e}')
        
        axes[1].scatter(acc_data, ood_data, c=c, marker='o')
        axes[1].set_xlabel(f'acc {acc_split}')
        axes[1].set_ylabel(f'{ood_metric} {label}')
        axes[1].set_title(f'Spearman r={corr_acc_ood:.2f}, p={p_acc_ood:.2e}')
        
        axes[2].scatter(nc_data, ood_data, c=c, marker='o')
        axes[2].set_xlabel(nc_metric)
        axes[2].set_ylabel(f'{ood_metric} {label}')
        axes[2].set_title(f'Spearman r={corr_nc_ood:.2f}, p={p_nc_ood:.2e}')
        
        plt.tight_layout()
        
        save_path = path.res_plots / benchmark_name
        save_path.mkdir(exist_ok=True, parents=True)
        filename = f'corr_{label}.png'
        print(str(save_path))
        plt.savefig(save_path / filename, bbox_inches='tight')
        plt.show()

    plot_corr_mean(2, 'nearood')
    plot_corr_mean(3, 'farood')

    def plot_corr_method(acc, nc, ood, ood_label, run_ids):
        for k in acc.keys():
            acc_data = np.array(acc[k])
            nc_data = np.array(nc[k])
            ood_data = np.array(ood[k])
            
            # Compute Spearman correlations
            corr_acc_nc, p_acc_nc = spearmanr(acc_data, nc_data)
            corr_acc_ood, p_acc_ood = spearmanr(acc_data, ood_data)
            corr_nc_ood, p_nc_ood = spearmanr(nc_data, ood_data)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f'{k} {ood_label}')
            
            c = [colors[i] for i in run_ids[k]]
            
            axes[0].scatter(acc_data, nc_data, c=c, marker='o')
            axes[0].set_xlabel(f'acc {acc_split}')
            axes[0].set_ylabel(nc_metric)
            axes[0].set_title(f'Spearman r={corr_acc_nc:.2f}, p={p_acc_nc:.2e}')
            
            axes[1].scatter(acc_data, ood_data, c=c, marker='o')
            axes[1].set_xlabel(f'acc {acc_split}')
            axes[1].set_ylabel(f'{ood_metric}')
            axes[1].set_title(f'Spearman r={corr_acc_ood:.2f}, p={p_acc_ood:.2e}')
            
            axes[2].scatter(nc_data, ood_data, c=c, marker='o')
            axes[2].set_xlabel(nc_metric)
            axes[2].set_ylabel(f'{ood_metric}')
            axes[2].set_title(f'Spearman r={corr_nc_ood:.2f}, p={p_nc_ood:.2e}')
            
            plt.tight_layout()
            
            save_path = path.res_plots / benchmark_name / f'corr_{ood_label}'
            save_path.mkdir(exist_ok=True, parents=True)
            filename = f'{k}.png'
            print(str(save_path))
            plt.savefig(save_path / filename, bbox_inches='tight')
            plt.show()

    plot_corr_method(acc, nc, nearood, 'nearood', run_ids)
    plot_corr_method(acc, nc, farood, 'farood', run_ids)

    def plot_corr_matrix():
        # Step 1: Prepare the data
        labels = ['acc', 'nc', 'nearood', 'farood']
        df = pd.DataFrame(data_mean, columns=labels)
        
        # Step 2: Compute Spearman correlation matrix
        spearman_corr = df.corr(method='spearman')
        
        # Step 3: Create a mask for the lower triangle (including the diagonal)
        mask = np.tril(np.ones_like(spearman_corr, dtype=bool), k=-1)
        
        # Step 4: Set up the matplotlib figure
        plt.figure(figsize=(10, 8))
        
        # Step 5: Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 20, as_cmap=True)
        
        # Step 6: Plot the heatmap
        sns.heatmap(
            spearman_corr,
            annot=True,
            fmt='.3f',
            cmap=cmap,
            mask=mask,
            square=True,
            cbar_kws={"shrink": 0.8, "label": "Spearman Correlation"},
            annot_kws={"size": 8, "color": "black"},
            linewidths=.5
        )
        
        plt.title('Spearman Correlation Matrix')
        plt.tight_layout()
        
        # Step 7: Save the figure
        save_path = path.res_plots / benchmark_name
        save_path.mkdir(exist_ok=True, parents=True)
        filename = f'corr_matrix_spearman.png'
        print(str(save_path))
        plt.savefig(save_path / filename, bbox_inches='tight')
        plt.show()

    plot_corr_matrix()


def plot_benchmark_specific(benchmark_name):
    plot_acc_nc_ood(benchmark_name)
    plot_acc_ood_avg(benchmark_name, acc_split='val', far=False, x_axis='acc')
    plot_acc_ood_avg(benchmark_name, acc_split='val', far=True, x_axis='acc')
    plot_acc_ood_avg(benchmark_name, acc_split='val', far=False, x_axis='epoch')
    plot_acc_ood_avg(benchmark_name, acc_split='val', far=True, x_axis='epoch')


if __name__ == '__main__':
    # cfg = OmegaConf.from_cli()
    
    # cfg.benchmark = 'cifar10'
    # cfg.run = 'run0'

    plot_noise()

    # if 'run' in cfg:
    #     plot_run_specific(cfg.benchmark, cfg.run)
    # else:
    #     plot_benchmark_specific(cfg.benchmark)

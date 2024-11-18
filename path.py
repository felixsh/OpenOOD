from pathlib import Path

# Loading
data_root = Path('/home/hauser/data_openood')
# data_root = Path('/storage_local/datasets/public/data_openood')

ckpt_root = Path('/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks')


# Saving
# res_root = Path('.')
res_root = Path('/mrtstorage/users/hauser/openood_res')

res_data = res_root / 'data'
res_data.mkdir(exist_ok=True, parents=True)

res_plots = res_root / 'res_plots'
res_plots.mkdir(exist_ok=True, parents=True)

res_scores = res_root / 'res_scores'
res_scores.mkdir(exist_ok=True, parents=True)

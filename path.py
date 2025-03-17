from pathlib import Path

# Loading
data_root = Path('/home/hauser/data_openood')
# data_root = Path('/storage_local/datasets/public/data_openood')

ckpt_root = Path('/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks')

torchvision_root = Path('/home/hauser/data')


# Saving
res_root = Path('/mrtstorage/users/hauser/openood_res')

res_db = Path('.') / 'db'
res_db.mkdir(exist_ok=True, parents=True)

res_data = res_root / 'data'
res_data.mkdir(exist_ok=True, parents=True)

res_plots = res_root / 'plots'
res_plots.mkdir(exist_ok=True, parents=True)

res_scores = res_root / 'scores'
res_scores.mkdir(exist_ok=True, parents=True)

cache_root = res_root / 'cache'
cache_root.mkdir(exist_ok=True, parents=True)

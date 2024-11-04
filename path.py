from pathlib import Path


res_data = Path('./res_data')
res_data.mkdir(exist_ok=True, parents=True)

res_plots = Path('./res_plots')
res_plots.mkdir(exist_ok=True, parents=True)

hdf5_data = Path('./hdf5_data')
hdf5_data.mkdir(exist_ok=True, parents=True)

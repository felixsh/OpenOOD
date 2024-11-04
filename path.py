from pathlib import Path


res_data = Path('./res_data')
res_data.mkdir(exist_ok=True, parents=True)

res_plots = Path('./res_plots')
res_plots.mkdir(exist_ok=True, parents=True)

res_scores = Path('./res_scores')
res_scores.mkdir(exist_ok=True, parents=True)
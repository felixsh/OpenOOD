import nc_toolbox as nctb
import pandas as pd
import torch
from tqdm import tqdm

from openood.evaluation_api.datasets import data_setup, get_id_ood_dataloader
from openood.evaluation_api.preprocessor import get_default_preprocessor
import path
from utils import load_network, get_batch_size


def eval_nc(benchmark_name, ckpt_path):
    
    # Parameters
    batch_size = get_batch_size(benchmark_name)
    shuffle = False
    num_workers = 8
    
    # Prepare stuff
    data_root = str(path.data_root)
    preprocessor = get_default_preprocessor(benchmark_name)
    net = load_network(benchmark_name, ckpt_path)

    # Load data
    data_setup(data_root, benchmark_name)
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers
    }
    dataloader_dict = get_id_ood_dataloader(benchmark_name, data_root,
                                                preprocessor, **loader_kwargs)
    dataloader = dataloader_dict['id']['train']

    # Get features and labels
    exampel_batch, _ = next(iter(dataloader))
    exampel_inp = exampel_batch['data'].cuda().float()

    _, exampel_feature = net(exampel_inp, return_feature=True)

    n = len(dataloader.dataset)
    d = exampel_feature.shape[1]

    H = torch.zeros((n, d), dtype=torch.float32, device='cuda')
    L = torch.zeros(n, dtype=int, device='cuda')

    idx = 0
    for batch in tqdm(dataloader):
        inp = batch['data'].cuda().float()
        _, feature = net(inp, return_feature=True)

        bs = inp.shape[0]
        H[idx : idx + bs] = feature.flatten(start_dim=1)
        L[idx : idx + bs] = batch['label'].cuda()
        idx += bs

    H = H.cpu().numpy()
    L = L.cpu().numpy()

    # Get weights and bias
    W, B = net.get_fc()  # (c x d), (c,)

    # Statistics
    mu_c = nctb.class_embedding_means(H, L)
    var_c = nctb.class_embedding_variances(H, L, mu_c)
    mu_g = nctb.global_embedding_mean(H)

    # NC metrics
    results = {
        "nc1_strong": nctb.nc1_strong(H, L, mu_c, mu_g),
        "nc1_weak": nctb.nc1_weak(H, L, mu_c, mu_g),
        "nc1_cdnv": nctb.nc1_cdnv(mu_c, var_c),
        "nc2_equinormness": nctb.nc2_equinormness(mu_c, mu_g),
        "nc2_equiangularity": nctb.nc2_equiangularity(mu_c, mu_g),
        "gnc2_hyperspherical_uniformity": nctb.gnc2_hypershperical_uniformity(mu_c, mu_g),
        "nc3_self_duality": nctb.nc3_self_duality(W, mu_c, mu_g),
        "unc3_uniform_duality": nctb.unc3_uniform_duality(W, mu_c, mu_g),
        "nc4_classifier_agreement": nctb.nc4_classifier_agreement(H, W, B, mu_c),
    }

    nc_metrics = pd.DataFrame(results)

    return nc_metrics

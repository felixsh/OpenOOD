import nc_toolbox as nctb
import pandas as pd


def eval_nc(feature_cache):

    split = 'train'
    H = feature_cache.get(split, 'features')
    L = feature_cache.get(split, 'labels')
    W = feature_cache.get(split, 'weights')
    B = feature_cache.get(split, 'bias')

    # Statistics
    mu_c = nctb.class_embedding_means(H, L)
    var_c = nctb.class_embedding_variances(H, L, mu_c)
    mu_g = nctb.global_embedding_mean(H)

    # NC metrics
    nc1_weak_between, nc1_weak_within = nctb.nc1_weak(H, L, mu_c, mu_g)
    results = {
        "nc1_strong": nctb.nc1_strong(H, L, mu_c, mu_g),
        "nc1_weak_between": nc1_weak_between,
        "nc1_weak_within": nc1_weak_within,
        "nc1_cdnv": nctb.nc1_cdnv(mu_c, var_c),
        "nc2_equinormness": nctb.nc2_equinormness(mu_c, mu_g),
        "nc2_equiangularity": nctb.nc2_equiangularity(mu_c, mu_g),
        "gnc2_hyperspherical_uniformity": nctb.gnc2_hypershperical_uniformity(mu_c, mu_g),
        "nc3_self_duality": nctb.nc3_self_duality(W, mu_c, mu_g),
        "unc3_uniform_duality": nctb.unc3_uniform_duality(W, mu_c, mu_g),
        "nc4_classifier_agreement": nctb.nc4_classifier_agreement(H, W, B, mu_c),
    }

    nc_metrics = pd.DataFrame([results])
    return nc_metrics

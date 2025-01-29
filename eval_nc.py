import nc_toolbox as nctb
import pandas as pd


def eval_nc(feature_cache, split='train'):

    # Features
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
        # "nc1_strong": nctb.nc1_strong(H, L, mu_c, mu_g),
        "nc1_weak_between": nc1_weak_between,
        "nc1_weak_within": nc1_weak_within,
        "nc1_cdnv_cov": nctb.nc1_cdnv(mu_c, var_c, reduction='cov'),
        "nc2_equinormness_cov": nctb.nc2_equinormness(mu_c, mu_g, reduction='cov'),
        "nc2_equiangularity_cov": nctb.nc2_equiangularity(mu_c, mu_g, reduction='cov'),
        "gnc2_hyperspherical_uniformity_cov": nctb.gnc2_hypershperical_uniformity(mu_c, mu_g, reduction='cov'),
        "nc3_self_duality": nctb.nc3_self_duality(W, mu_c, mu_g),
        "unc3_uniform_duality_cov": nctb.unc3_uniform_duality(W, mu_c, mu_g, reduction='cov'),
        "nc4_classifier_agreement": nctb.nc4_classifier_agreement(H, W, B, mu_c),

        "nc1_cdnv_mean": nctb.nc1_cdnv(mu_c, var_c, reduction='mean'),
        "nc2_equinormness_mean": nctb.nc2_equinormness(mu_c, mu_g, reduction='mean'),
        "nc2_equiangularity_mean": nctb.nc2_equiangularity(mu_c, mu_g, reduction='mean'),
        "gnc2_hyperspherical_uniformity_mean": nctb.gnc2_hypershperical_uniformity(mu_c, mu_g, reduction='mean'),
        "unc3_uniform_duality_mean": nctb.unc3_uniform_duality(W, mu_c, mu_g, reduction='mean'),
    }

    nc_metrics = pd.DataFrame([results])
    return nc_metrics

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import path
from utils import extract_datetime_from_path

DB_NAME = path.res_db / 'results.db'


def _create_db() -> None:
    with sqlite3.connect(DB_NAME) as dbconn:
        dbconn.execute("""
        CREATE TABLE IF NOT EXISTS acc (
            benchmark TEXT NOT NULL,
            model TEXT NOT NULL,
            run TEXT NOT NULL,
            epoch INTEGER NOT NULL,
            dataset TEXT NOT NULL,
            split TEXT,
            value REAL NOT NULL,
            UNIQUE (benchmark, model, run, epoch, dataset, split) ON CONFLICT REPLACE
        )
        """)

        dbconn.execute("""
        CREATE TABLE IF NOT EXISTS ood (
            benchmark TEXT NOT NULL,
            model TEXT NOT NULL,
            run TEXT NOT NULL,
            epoch INTEGER NOT NULL,
            method TEXT NOT NULL,
            dataset TEXT NOT NULL,
            FPRat95 REAL NOT NULL,
            AUROC REAL NOT NULL,
            AUPR_IN REAL NOT NULL,
            AUPR_OUT REAL NOT NULL,
            UNIQUE (benchmark, model, run, epoch, method, dataset) ON CONFLICT REPLACE
        )
        """)

        dbconn.execute("""
        CREATE TABLE IF NOT EXISTS nc (
            benchmark TEXT NOT NULL,
            model TEXT NOT NULL,
            run TEXT NOT NULL,
            epoch INTEGER NOT NULL,
            dataset TEXT NOT NULL,
            split TEXT,
            nc1_strong REAL,
            nc1_weak_between REAL,
            nc1_weak_within REAL,
            nc1_cdnv_cov REAL,
            nc2_equinormness_cov REAL,
            nc2_equiangularity_cov REAL,
            gnc2_hyperspherical_uniformity_cov REAL,
            nc3_self_duality REAL,
            unc3_uniform_duality_cov REAL,
            nc4_classifier_agreement REAL,
            nc1_cdnv_mean REAL,
            nc2_equinormness_mean REAL,
            nc2_equiangularity_mean REAL,
            gnc2_hyperspherical_uniformity_mean REAL,
            unc3_uniform_duality_mean REAL,
            UNIQUE (benchmark, model, run, epoch, dataset, split) ON CONFLICT REPLACE
        )
        """)

    dbconn.close()


def store_acc(
    benchmark: str,
    model: str,
    run: str,
    epoch: int,
    dataset: str,
    split: str,
    acc: float,
) -> None:
    query = """
        INSERT INTO acc (benchmark, model, run, epoch, dataset, split, value)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """
    data = (benchmark, model, run, epoch, dataset, split, acc)

    with sqlite3.connect(DB_NAME) as dbconn:
        dbconn.execute(query, data)

    dbconn.close()


def unpack_ood(df: pd.DataFrame) -> Iterator[tuple[str]]:
    for row_name, row in df.iterrows():
        yield row_name, row['FPR@95'], row['AUROC'], row['AUPR_IN'], row['AUPR_OUT']


def store_ood(
    benchmark: str, model: str, run: str, epoch: int, ood_method: str, df: pd.DataFrame
) -> None:
    query = """
        INSERT INTO ood (benchmark, model, run, epoch, method, dataset, FPRat95, AUROC, AUPR_IN, AUPR_OUT)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """

    data = ((benchmark, model, run, epoch, ood_method, *tup) for tup in unpack_ood(df))

    with sqlite3.connect(DB_NAME) as dbconn:
        dbconn.executemany(query, data)

    dbconn.close()


nc_keys = [
    'nc1_strong',
    'nc1_weak_between',
    'nc1_weak_within',
    'nc1_cdnv_cov',
    'nc2_equinormness_cov',
    'nc2_equiangularity_cov',
    'gnc2_hyperspherical_uniformity_cov',
    'nc3_self_duality',
    'unc3_uniform_duality_cov',
    'nc4_classifier_agreement',
    'nc1_cdnv_mean',
    'nc2_equinormness_mean',
    'nc2_equiangularity_mean',
    'gnc2_hyperspherical_uniformity_mean',
    'unc3_uniform_duality_mean',
]


def unpack_nc(df: pd.DataFrame) -> list[float]:
    # All keys present in dict
    res = {key: None for key in nc_keys}

    # Insert values
    for column_name, series in df.items():
        res[column_name] = series.iloc[0]

    # Convert to tuple
    return list(res.values())


def store_nc(
    benchmark: str,
    model: str,
    run: str,
    epoch: int,
    dataset: str,
    split: str,
    df: pd.DataFrame,
) -> None:
    query = """
    INSERT INTO nc (
    benchmark,
    model,
    run,
    epoch,
    dataset,
    split,
    nc1_strong,
    nc1_weak_between,
    nc1_weak_within,
    nc1_cdnv_cov,
    nc2_equinormness_cov,
    nc2_equiangularity_cov,
    gnc2_hyperspherical_uniformity_cov,
    nc3_self_duality,
    unc3_uniform_duality_cov,
    nc4_classifier_agreement,
    nc1_cdnv_mean,
    nc2_equinormness_mean,
    nc2_equiangularity_mean,
    gnc2_hyperspherical_uniformity_mean,
    unc3_uniform_duality_mean
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """
    df_flat = unpack_nc(df)
    data = (benchmark, model, run, epoch, dataset, split, *df_flat)

    with sqlite3.connect(DB_NAME) as dbconn:
        dbconn.execute(query, data)

    dbconn.close()


def store_hdf5(hf5_path: Path) -> None:
    benchmark = str(hf5_path.relative_to(path.res_data).parents[-2])
    model = str(hf5_path.relative_to(path.res_data).parents[-3].stem)
    run = extract_datetime_from_path(hf5_path)
    epoch = int(hf5_path.stem[1:])

    if (
        'resnet18' in model.lower()
        or 'alexnet' in model.lower()
        or 'mobilenet' in model.lower()
        or 'vgg' in model.lower()
    ):
        dataset = 'cifar10'
    else:
        dataset = benchmark

    with pd.HDFStore(hf5_path, mode='r') as store:
        ood_keys = list(store.keys())
        try:
            ood_keys.remove('/nc')
        except ValueError:
            pass

        try:
            df = store.get('/nc_train')
            store_nc(benchmark, model, run, epoch, dataset, 'train', df)
            ood_keys.remove('/nc_train')
        except KeyError:
            print(hf5_path)
            print(ood_keys)
            input()

        try:
            df = store.get('/nc_val')
            store_nc(benchmark, model, run, epoch, dataset, 'val', df)
            ood_keys.remove('/nc_val')
        except KeyError:
            print(hf5_path)
            print(ood_keys)
            input()

        try:
            df = store.get('/acc')
            store_acc(
                benchmark, model, run, epoch, dataset, 'train', df.at['id', 'train']
            )
            ood_keys.remove('/acc')
        except KeyError:
            print(hf5_path)
            print(ood_keys)
            input()

        for k in ood_keys:
            df = store.get(k)
            key = k[1:]
            store_ood(benchmark, model, run, epoch, key, df)


def fill_db_with_previous_results(dirs: list[Path]) -> None:
    h5_paths = [h5_path for d in dirs for h5_path in Path(d).rglob('e*.h5')]
    for h5_path in tqdm(h5_paths):
        store_hdf5(h5_path)


def count_rows() -> None:
    with sqlite3.connect(DB_NAME) as dbconn:
        n_acc = dbconn.execute('SELECT COUNT(*) FROM acc;').fetchone()[0]
        n_nc = dbconn.execute('SELECT COUNT(*) FROM nc;').fetchone()[0]
        n_ood = dbconn.execute('SELECT COUNT(*) FROM ood;').fetchone()[0]
    dbconn.close()

    print(f'ACC:\t{n_acc}')
    print(f'NC:\t{n_nc}')
    print(f'OOD:\t{n_ood}')


def find_mismatched_keys() -> tuple[pd.DataFrame, pd.DataFrame]:
    conn = sqlite3.connect(DB_NAME)

    # Query 1: In acc but not in nc
    query_acc_not_nc = """
    SELECT acc.benchmark, acc.model, acc.run, acc.epoch, acc.dataset, acc.split
    FROM acc
    LEFT JOIN nc ON acc.benchmark = nc.benchmark
                AND acc.model = nc.model
                AND acc.run = nc.run
                AND acc.epoch = nc.epoch
                AND acc.dataset = nc.dataset
                AND acc.split = nc.split
    WHERE nc.benchmark IS NULL;
    """
    acc_not_nc_df = pd.read_sql_query(query_acc_not_nc, conn)

    # Query 2: In nc but not in acc
    query_nc_not_acc = """
    SELECT nc.benchmark, nc.model, nc.run, nc.epoch, nc.dataset, nc.split
    FROM nc
    LEFT JOIN acc ON nc.benchmark = acc.benchmark
                AND nc.model = acc.model
                AND nc.run = acc.run
                AND nc.epoch = acc.epoch
                AND nc.dataset = acc.dataset
                AND nc.split = acc.split
    WHERE acc.benchmark IS NULL;
    """
    nc_not_acc_df = pd.read_sql_query(query_nc_not_acc, conn)

    conn.close()

    return acc_not_nc_df, nc_not_acc_df


def key_exists_acc(
    benchmark: str, model: str, run: str, epoch: int, dataset: str, split: str
) -> bool:
    query = """
    SELECT 1 FROM acc
    WHERE benchmark = ?
      AND model = ?
      AND run = ?
      AND epoch = ?
      AND dataset = ?
      AND split = ?
    LIMIT 1;
    """

    with sqlite3.connect(DB_NAME) as conn:
        result = conn.execute(
            query, (benchmark, model, run, epoch, dataset, split)
        ).fetchone()

    return result is not None


def key_exists_nc(
    benchmark: str, model: str, run: str, epoch: int, dataset: str, split: str
) -> bool:
    query = """
    SELECT 1 FROM nc
    WHERE benchmark = ?
      AND model = ?
      AND run = ?
      AND epoch = ?
      AND dataset = ?
      AND split = ?
    LIMIT 1;
    """

    with sqlite3.connect(DB_NAME) as conn:
        result = conn.execute(
            query, (benchmark, model, run, epoch, dataset, split)
        ).fetchone()

    return result is not None


def key_exists_ood(
    benchmark: str, model: str, run: str, epoch: int, method: str
) -> bool:
    query = """
    SELECT 1 FROM ood
    WHERE benchmark = ?
      AND model = ?
      AND run = ?
      AND epoch = ?
      AND method = ?
    LIMIT 1;
    """

    with sqlite3.connect(DB_NAME) as conn:
        result = conn.execute(query, (benchmark, model, run, epoch, method)).fetchone()

    return result is not None


if __name__ == '__main__':
    # _create_db()

    # p = Path(
    #     '/mrtstorage/users/hauser/openood_res/data/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/e1.h5'
    # )
    # store_hdf5(p)

    benchmark_dirs = [
        '/mrtstorage/users/hauser/openood_res/data/cifar10/ResNet18_32x32/no_noise/300+_epochs',
        '/mrtstorage/users/hauser/openood_res/data/cifar100/type/no_noise/1000+_epochs/',
        '/mrtstorage/users/hauser/openood_res/data/imagenet200/type/no_noise/1000+_epochs/',
        '/mrtstorage/users/hauser/openood_res/data/imagenet/ResNet50/no_noise/150+_epochs',
        '/mrtstorage/users/hauser/openood_res/data/cifar10/NCAlexNet/no_noise/300+_epochs',
        '/mrtstorage/users/hauser/openood_res/data/cifar10/NCMobileNetV2/no_noise/300+_epochs',
        '/mrtstorage/users/hauser/openood_res/data/cifar10/NCVGG16/no_noise/300+_epochs',
    ]

    # fill_db_with_previous_results(benchmark_dirs)

    # count_rows()

    acc_not_nc_df, nc_not_acc_df = find_mismatched_keys()

    pd.set_option('display.width', None)
    print('Keys in acc but not in nc:')
    print(acc_not_nc_df)
    print('\nKeys in nc but not in acc:')
    print(nc_not_acc_df)

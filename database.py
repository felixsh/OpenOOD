import io
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import path
import utils

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


def _create_stats_table() -> None:
    with sqlite3.connect(DB_NAME) as dbconn:
        dbconn.execute("""
        CREATE TABLE IF NOT EXISTS stats (
            benchmark TEXT NOT NULL,
            model TEXT NOT NULL,
            run TEXT NOT NULL,
            epoch INTEGER NOT NULL,
            dataset TEXT NOT NULL,
            split TEXT,
            n_class INT,
            mu_g array,
            var_g REAL,
            mu_c array,
            var_c array,
            UNIQUE (benchmark, model, run, epoch, dataset, split) ON CONFLICT REPLACE
        )
        """)
    dbconn.close()


def add_new_column(
    table: str,
    column: str,
    column_type: str = 'TEXT',
    conn: sqlite3.Connection | None = None,
) -> None:
    """Adds the new column to table if it doesn't already exist."""
    try:
        conn_supplied = conn is None
        if not conn_supplied:
            conn = sqlite3.connect(DB_NAME)
        conn.execute(f'ALTER TABLE {table} ADD COLUMN {column} {column_type};')
        if not conn_supplied:
            conn.close()
    except sqlite3.OperationalError as e:
        if 'duplicate column name' in str(e).lower():
            print("Column 'hyperparameter' already exists in 'ood' table.")
        else:
            raise


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
    benchmark: str,
    model: str,
    run: str,
    epoch: int,
    ood_method: str,
    df: pd.DataFrame,
    hyperparams: int | float | tuple[int, float] | None,
) -> None:
    query = """
        INSERT INTO ood (benchmark, model, run, epoch, method, dataset, FPRat95, AUROC, AUPR_IN, AUPR_OUT, hyperparameter)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """

    hyperparams_str = str(hyperparams) if hyperparams is not None else None
    data = (
        (benchmark, model, run, epoch, ood_method, *tup, hyperparams_str)
        for tup in unpack_ood(df)
    )

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


def store_stats(
    benchmark: str,
    model: str,
    run: str,
    epoch: int,
    dataset: str,
    split: str,
    n_class: int,
    mu_g: np.ndarray,
    var_g: float,
    mu_c: np.ndarray,
    var_c: np.ndarray,
) -> None:
    query = """
        INSERT INTO stats (
        benchmark,
        model,
        run,
        epoch,
        dataset,
        split,
        n_class,
        mu_g,
        var_g,
        mu_c,
        var_c
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
    data = (
        benchmark,
        model,
        run,
        epoch,
        dataset,
        split,
        n_class,
        mu_g,
        var_g,
        mu_c,
        var_c,
    )

    def adapt_array(arr):
        out = io.BytesIO()
        np.save(out, arr)
        return sqlite3.Binary(out.getvalue())

    sqlite3.register_adapter(np.ndarray, adapt_array)
    sqlite3.register_converter('array', lambda x: np.load(io.BytesIO(x)))

    with sqlite3.connect(DB_NAME, detect_types=sqlite3.PARSE_DECLTYPES) as dbconn:
        dbconn.execute(query, data)

    dbconn.close()


def transfer_hdf5(hf5_path: Path) -> None:
    benchmark = str(hf5_path.relative_to(path.res_data).parents[-2])
    model = str(hf5_path.relative_to(path.res_data).parents[-3].stem)
    run = utils.extract_datetime_from_path(hf5_path)
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


def transfer_previous_results(dirs: list[Path]) -> None:
    h5_paths = [h5_path for d in dirs for h5_path in Path(d).rglob('e*.h5')]
    for h5_path in tqdm(h5_paths):
        transfer_hdf5(h5_path)


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


def update_ood_hyperparameter(
    benchmark: str,
    model: str,
    run: str,
    epoch: int,
    method: str,
    hyperparameter: int | float | tuple[int, float] | None,
) -> None:
    """
    Updates the 'hyperparameter' column for all matching rows in the ood table,
    ignoring the dataset.

    :param hyperparameter: Value to store. Can be int, float, tuple[int, float], or None.
    """
    value_str = str(hyperparameter) if hyperparameter is not None else None

    query = """
    UPDATE ood
    SET hyperparameter = ?
    WHERE benchmark = ?
      AND model = ?
      AND run = ?
      AND epoch = ?
      AND method = ?;
    """

    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.execute(query, (value_str, benchmark, model, run, epoch, method))
        conn.commit()

        if cursor.rowcount == 0:
            print('Warning: No matching rows found to update.')
        else:
            print(f'Updated {cursor.rowcount} row(s).')

    conn.close()


def transfer_hyperparams() -> None:
    filename = 'hyperparam.log'
    with open(filename, 'r') as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(',')]
            if len(parts) < 3:
                continue

            benchmark = parts[0]
            method = parts[1]
            *middle, last = parts[2:]

            hyperparams_str = ', '.join(middle)
            ckpt_path = Path(last)

            model_name = utils.get_model_name(ckpt_path)
            run_id = utils.extract_datetime_from_path(ckpt_path)
            epoch = utils.get_epoch_number(ckpt_path)

            # print(benchmark, model_name, run_id, epoch, method, hyperparams_str)

            update_ood_hyperparameter(
                benchmark, model_name, run_id, epoch, method, hyperparams_str
            )


def replace_value_in_tables(
    column: str, old_val: str, new_val: str, tables: list[str] = ['acc', 'nc', 'ood']
) -> None:
    """
    Replaces all occurrences of a specific column value in all target tables.

    :param column: The column name to update.
    :param old_val: The value to replace.
    :param new_val: The new value to insert.
    :param tables: List of table names to apply the update (default: all three).
    """
    with sqlite3.connect(DB_NAME) as conn:
        for table in tables:
            query = f"""
            UPDATE {table}
            SET {column} = ?
            WHERE {column} = ?;
            """
            try:
                cursor = conn.execute(query, (new_val, old_val))
                print(
                    f"[{table}] Updated {cursor.rowcount} row(s) where {column} = '{old_val}'."
                )
            except sqlite3.OperationalError as e:
                print(f'[{table}] Skipped: {e}')
        conn.commit()
    conn.close()


def export_run_csv(
    csv_path: str = 'all_runs.csv',
    placeholder: str | None = None,
) -> None:
    conn = sqlite3.connect(DB_NAME)

    combined_df = pd.DataFrame()

    for table in ['acc', 'nc', 'ood']:
        column_info = pd.read_sql_query(f'PRAGMA table_info({table});', conn)
        has_experiment = 'experiment' in column_info['name'].values

        base_cols = (
            'benchmark, model, run, MIN(epoch) as min_epoch, MAX(epoch) as max_epoch'
        )
        query = f"""
        SELECT {base_cols}{', experiment' if has_experiment else ''}
        FROM {table}
        GROUP BY benchmark, model, run
        """
        df = pd.read_sql_query(query, conn)

        # Fill missing experiment values
        if 'experiment' not in df.columns:
            df['experiment'] = pd.Series(dtype='string')

        combined_df = pd.concat([combined_df, df], ignore_index=True)

    conn.close()

    combined_df.drop_duplicates(
        subset=['benchmark', 'model', 'run', 'experiment'], inplace=True
    )
    combined_df.to_csv(csv_path, index=False)


def apply_run_csv(csv_path: str = 'all_runs.csv') -> None:
    """
    Reads a CSV containing (benchmark, model, run, experiment) and updates the experiment
    column for matching rows in acc, nc, and ood tables.

    If the experiment column doesn't exist, it is added.
    """
    df = pd.read_csv(csv_path)

    required_cols = {'benchmark', 'model', 'run', 'experiment'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f'CSV must contain columns: {required_cols}')

    df = df.dropna(subset=['experiment'])  # Only keep rows where experiment is filled

    with sqlite3.connect(DB_NAME) as conn:
        for table in ['acc', 'nc', 'ood']:
            add_new_column(table, column='experiment', column_type='TEXT', conn=conn)

            for _, row in df.iterrows():
                benchmark = row['benchmark']
                model = row['model']
                run = row['run']
                experiment = row['experiment']

                query = f"""
                UPDATE {table}
                SET experiment = ?
                WHERE benchmark = ? AND model = ? AND run = ?;
                """
                cursor = conn.execute(query, (experiment, benchmark, model, run))
                print(
                    f"[{table}] Set experiment='{experiment}' for {cursor.rowcount} row(s) "
                    f'({benchmark}, {model}, {run})'
                )

            conn.commit()
    conn.close()


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

    # acc_not_nc_df, nc_not_acc_df = find_mismatched_keys()
    # pd.set_option('display.width', None)
    # print('Keys in acc but not in nc:')
    # print(acc_not_nc_df)
    # print('\nKeys in nc but not in acc:')
    # print(nc_not_acc_df)

    # add_new_column(table='ood', column='hyperparameter', column_type='TEXT')
    # transfer_hyperparams()

    # replace_value_in_tables('model', 'type', 'NCResNet18_32x32')

    # _create_stats_table()

    # export_run_csv()
    # apply_run_csv()

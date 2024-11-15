import os
import pandas as pd
from pathlib import Path


def add_hdf5_and_md_to_target(source_dir, target_dir):
    # Walk through subdirectories in source and target directories
    for src_subdir, tgt_subdir in zip(sorted(Path(source_dir).rglob('*')), sorted(Path(target_dir).rglob('*'))):
        # Check if both are directories
        if src_subdir.is_dir() and tgt_subdir.is_dir():
            # Process HDF5 files
            for src_hdf5_file in src_subdir.glob("*.h5"):
                tgt_hdf5_file = tgt_subdir / src_hdf5_file.name
                with pd.HDFStore(tgt_hdf5_file, mode='a') as tgt_store:
                    with pd.HDFStore(src_hdf5_file, mode='r') as src_store:
                        for key in ['/dice', '/msp', '/odin']:
                            data = src_store[key]
                            tgt_store.put(key, data)
                            print(f"Added {key} from {src_hdf5_file} to {tgt_hdf5_file}")

            # Process TXT files
            for src_txt_file in src_subdir.glob("*.md"):
                tgt_md_file = tgt_subdir / src_txt_file.name
                with open(src_txt_file, "r") as txt_file, open(tgt_md_file, "a") as md_file:
                    content = txt_file.read()
                    md_file.write(content + "\n\n")  # Append with extra line breaks
                    print(f"Appended content from {src_txt_file} to {tgt_md_file}")


if __name__ == '__main__':
    source_dir = '/mrtstorage/users/hauser/imagenet/run0'
    target_dir = '/home/hauser/OpenOOD/res_data/imagenet/run0'
    add_hdf5_and_md_to_target(source_dir, target_dir)

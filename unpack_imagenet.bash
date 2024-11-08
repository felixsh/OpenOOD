#!/bin/bash

# Set the source directory and base unpacking path
SOURCE_DIR="/mrtstorage/datasets_tmp/ImageNet2012/unpacked"
BASE_PATH="/mrtstorage/datasets_tmp/ImageNet2012/imagenet_1k/train/"

# Create the base directory if it doesn't exist
mkdir -p "$BASE_PATH"

# Loop over each tar file in the SOURCE_DIR
for tarfile in "$SOURCE_DIR"/*.tar "$SOURCE_DIR"/*.tar.gz "$SOURCE_DIR"/*.tgz; do
  # Check if any files match the glob pattern
  if [ -f "$tarfile" ]; then
    # Get the filename without path
    filename=$(basename "$tarfile")
    
    # Remove .tar or .tar.gz or .tgz extension to create the directory name
    dirname="${filename%.*}"
    dirname="${dirname%.*}"  # For .tar.gz cases

    # Create the target directory within BASE_PATH
    target_dir="$BASE_PATH$dirname"
    mkdir -p "$target_dir"

    # Unpack the tar file into the target directory
    tar -xf "$tarfile" -C "$target_dir"
    echo "Unpacked $tarfile into $target_dir/"
  else
    echo "No tar files found in $SOURCE_DIR."
    break
  fi
done
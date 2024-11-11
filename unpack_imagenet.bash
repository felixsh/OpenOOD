#!/bin/bash

# Set paths within imagenet_1k directory
IMAGENET_DIR="./imagenet_1k"
TMP_DIR="$IMAGENET_DIR/tmp_unpack_dir"
TRAIN_DIR="$IMAGENET_DIR/train"

# Create necessary directories
mkdir -p "$TRAIN_DIR"

# Check if the initial tar file exists in the imagenet_1k directory
if [ -f "$IMAGENET_DIR/ILSVRC2012_img_train.tar" ]; then
  # Create a temporary directory for initial unpacking
  mkdir -p "$TMP_DIR"
  
  # Unpack ILSVRC2012_img_train.tar into the temporary directory
  tar -xf "$IMAGENET_DIR/ILSVRC2012_img_train.tar" -C "$TMP_DIR"
  echo "Unpacked ILSVRC2012_img_train.tar into $TMP_DIR/"
else
  echo "ILSVRC2012_img_train.tar not found in $IMAGENET_DIR. Exiting."
  exit 1
fi

# Loop over each tar file in the temporary directory
for tarfile in "$TMP_DIR"/*.tar "$TMP_DIR"/*.tar.gz "$TMP_DIR"/*.tgz; do
  # Check if any files match the glob pattern
  if [ -f "$tarfile" ]; then
    # Get the filename without path
    filename=$(basename "$tarfile")
    
    # Remove .tar or .tar.gz or .tgz extension to create the directory name
    dirname="${filename%.*}"
    dirname="${dirname%.*}"  # For .tar.gz cases

    # Create the target directory within TRAIN_DIR
    target_dir="$TRAIN_DIR/$dirname"
    mkdir -p "$target_dir"

    # Unpack the tar file into the target directory
    tar -xf "$tarfile" -C "$target_dir"
    echo "Unpacked $tarfile into $target_dir/"
  else
    echo "No additional tar files found in $TMP_DIR."
    break
  fi
done

# Clean up: remove the temporary directory
rm -rf "$TMP_DIR"
echo "Temporary directory $TMP_DIR deleted."

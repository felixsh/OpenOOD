#!/bin/bash

# msp odin mds react dice knn nusa vim ncscore neco epa

trap 'pkill -P $$; exit' SIGINT SIGTERM

OOD="[msp,odin,mds,react,dice,knn,nusa,vim,ncscore,neco,epa]"
PARENT_DIR=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/

# Get all subdirectories, sort them, and store in an array
subdirs=($(find "$PARENT_DIR" -mindepth 1 -maxdepth 1 -type d | sort))

# Loop over subdirectories in batches of four
batch_size=4
total=${#subdirs[@]}
echo $total

for ((i = 0; i < total; i += batch_size)); do

    batch=("${subdirs[@]:i:batch_size}")

    CUDA_VISIBLE_DEVICES=4 python eval_main.py run="${batch[0]}" ood=$OOD &
    CUDA_VISIBLE_DEVICES=5 python eval_main.py run="${batch[1]}" ood=$OOD &
    CUDA_VISIBLE_DEVICES=6 python eval_main.py run="${batch[2]}" ood=$OOD &
    CUDA_VISIBLE_DEVICES=7 python eval_main.py run="${batch[3]}" ood=$OOD &

    wait $(jobs -p)
done
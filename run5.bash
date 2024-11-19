#!/bin/bash

# msp odin mds react dice knn nusa vim ncscore neco epa

trap 'pkill -P $$; exit' SIGINT SIGTERM

RUN0=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53
RUN1=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56
OOD="[msp,odin,mds,react,dice,knn,nusa,vim,ncscore,neco,epa]"

CUDA_VISIBLE_DEVICES=3 python eval_main.py run=$RUN0 ood=$OOD &
CUDA_VISIBLE_DEVICES=4 python eval_main.py run=$RUN1 ood=$OOD &
wait $(jobs -p)

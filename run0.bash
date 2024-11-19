#!/bin/bash

# msp odin mds react dice knn nusa vim ncscore neco epa

trap 'pkill -P $$; exit' SIGINT SIGTERM

RUN=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/
OOD="[msp,odin,mds,react,dice,knn,nusa,vim,ncscore,neco,epa]"

CUDA_VISIBLE_DEVICES=3 python eval_main.py run=$RUN ood=$OOD
wait $(jobs -p)

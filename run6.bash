#!/bin/bash

# msp odin mds react dice knn nusa vim ncscore neco epa

trap 'pkill -P $$; exit' SIGINT SIGTERM

RUN0=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59
RUN1=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57
RUN2=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50
RUN3=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40

OOD="[msp,odin,mds,react,dice,knn,nusa,vim,ncscore,neco,epa]"

CUDA_VISIBLE_DEVICES=0 python eval_main.py run=$RUN0 ood=$OOD &
CUDA_VISIBLE_DEVICES=1 python eval_main.py run=$RUN1 ood=$OOD &
CUDA_VISIBLE_DEVICES=2 python eval_main.py run=$RUN2 ood=$OOD &
CUDA_VISIBLE_DEVICES=3 python eval_main.py run=$RUN3 ood=$OOD &
wait $(jobs -p)

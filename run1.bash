#!/bin/bash

trap 'pkill -P $$; exit' SIGINT SIGTERM

BENCHMARK=imagenet200

CUDA_VISIBLE_DEVICES=0 python main_ood.py benchmark=$BENCHMARK postprocessor=msp &
CUDA_VISIBLE_DEVICES=1 python main_ood.py benchmark=$BENCHMARK postprocessor=mds &
CUDA_VISIBLE_DEVICES=2 python main_ood.py benchmark=$BENCHMARK postprocessor=odin &
CUDA_VISIBLE_DEVICES=3 python main_ood.py benchmark=$BENCHMARK postprocessor=knn &
wait $(jobs -p)

CUDA_VISIBLE_DEVICES=0 python main_ood.py benchmark=$BENCHMARK postprocessor=react &
CUDA_VISIBLE_DEVICES=1 python main_ood.py benchmark=$BENCHMARK postprocessor=dice &
CUDA_VISIBLE_DEVICES=2 python main_ood.py benchmark=$BENCHMARK postprocessor=nusa &
CUDA_VISIBLE_DEVICES=3 python main_ood.py benchmark=$BENCHMARK postprocessor=vim &
wait $(jobs -p)

CUDA_VISIBLE_DEVICES=0 python main_ood.py benchmark=$BENCHMARK postprocessor=ncscore &
CUDA_VISIBLE_DEVICES=1 python main_ood.py benchmark=$BENCHMARK postprocessor=neco &
CUDA_VISIBLE_DEVICES=2 python main_ood.py benchmark=$BENCHMARK postprocessor=epa &
wait $(jobs -p)


BENCHMARK=imagenet1k

CUDA_VISIBLE_DEVICES=0 python main_ood.py benchmark=$BENCHMARK postprocessor=msp &
CUDA_VISIBLE_DEVICES=1 python main_ood.py benchmark=$BENCHMARK postprocessor=mds &
CUDA_VISIBLE_DEVICES=2 python main_ood.py benchmark=$BENCHMARK postprocessor=odin &
CUDA_VISIBLE_DEVICES=3 python main_ood.py benchmark=$BENCHMARK postprocessor=knn &
wait $(jobs -p)

CUDA_VISIBLE_DEVICES=0 python main_ood.py benchmark=$BENCHMARK postprocessor=react &
CUDA_VISIBLE_DEVICES=1 python main_ood.py benchmark=$BENCHMARK postprocessor=dice &
CUDA_VISIBLE_DEVICES=2 python main_ood.py benchmark=$BENCHMARK postprocessor=nusa &
CUDA_VISIBLE_DEVICES=3 python main_ood.py benchmark=$BENCHMARK postprocessor=vim &
wait $(jobs -p)

CUDA_VISIBLE_DEVICES=0 python main_ood.py benchmark=$BENCHMARK postprocessor=ncscore &
CUDA_VISIBLE_DEVICES=1 python main_ood.py benchmark=$BENCHMARK postprocessor=neco &
CUDA_VISIBLE_DEVICES=2 python main_ood.py benchmark=$BENCHMARK postprocessor=epa &
wait $(jobs -p)
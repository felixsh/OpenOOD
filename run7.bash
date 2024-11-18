#!/bin/bash

# msp odin mds react dice knn nusa vim ncscore neco epa

trap 'pkill -P $$; exit' SIGINT SIGTERM

BENCHMARK=cifar10_noise

#CUDA_VISIBLE_DEVICES=0 python eval_main.py benchmark=$BENCHMARK run=run0 &
#CUDA_VISIBLE_DEVICES=1 python eval_main.py benchmark=$BENCHMARK run=run1 &
#CUDA_VISIBLE_DEVICES=2 python eval_main.py benchmark=$BENCHMARK run=run2 &
#wait $(jobs -p)

CUDA_VISIBLE_DEVICES=0 python eval_main.py benchmark=$BENCHMARK run=run0 &
CUDA_VISIBLE_DEVICES=1 python eval_main.py benchmark=$BENCHMARK run=run1 &
CUDA_VISIBLE_DEVICES=2 python eval_main.py benchmark=$BENCHMARK run=run2 &
CUDA_VISIBLE_DEVICES=3 python eval_main.py benchmark=$BENCHMARK run=run3 &
CUDA_VISIBLE_DEVICES=4 python eval_main.py benchmark=$BENCHMARK run=run4 &
CUDA_VISIBLE_DEVICES=5 python eval_main.py benchmark=$BENCHMARK run=run5 &
CUDA_VISIBLE_DEVICES=6 python eval_main.py benchmark=$BENCHMARK run=run6 &
CUDA_VISIBLE_DEVICES=7 python eval_main.py benchmark=$BENCHMARK run=run7 &
wait $(jobs -p)

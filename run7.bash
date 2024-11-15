#!/bin/bash

# msp odin mds react dice knn nusa vim ncscore neco epa

trap 'pkill -P $$; exit' SIGINT SIGTERM

BENCHMARK=cifar10_noise

CUDA_VISIBLE_DEVICES=0 python eval_main.py benchmark=$BENCHMARK run=run0 &
CUDA_VISIBLE_DEVICES=1 python eval_main.py benchmark=$BENCHMARK run=run1 &
CUDA_VISIBLE_DEVICES=2 python eval_main.py benchmark=$BENCHMARK run=run2 &
wait $(jobs -p)

CUDA_VISIBLE_DEVICES=0 python eval_main.py benchmark=$BENCHMARK run=run3 &
CUDA_VISIBLE_DEVICES=1 python eval_main.py benchmark=$BENCHMARK run=run4 &
CUDA_VISIBLE_DEVICES=2 python eval_main.py benchmark=$BENCHMARK run=run5 &
wait $(jobs -p)

CUDA_VISIBLE_DEVICES=0 python eval_main.py benchmark=$BENCHMARK run=run6 &
CUDA_VISIBLE_DEVICES=1 python eval_main.py benchmark=$BENCHMARK run=run7 &
CUDA_VISIBLE_DEVICES=2 python eval_main.py benchmark=$BENCHMARK run=run8 &
wait $(jobs -p)

CUDA_VISIBLE_DEVICES=0 python eval_main.py benchmark=$BENCHMARK run=run9 &
CUDA_VISIBLE_DEVICES=1 python eval_main.py benchmark=$BENCHMARK run=run10 &
CUDA_VISIBLE_DEVICES=2 python eval_main.py benchmark=$BENCHMARK run=run11 &
wait $(jobs -p)

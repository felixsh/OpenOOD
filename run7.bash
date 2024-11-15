#!/bin/bash

# msp odin mds react dice knn nusa vim ncscore neco epa

trap 'pkill -P $$; exit' SIGINT SIGTERM

BENCHMARK=imagenet
RUN=run0

CUDA_VISIBLE_DEVICES=0 python eval_main.py benchmark=$BENCHMARK run=$RUN epoch=1 &
CUDA_VISIBLE_DEVICES=1 python eval_main.py benchmark=$BENCHMARK run=$RUN epoch=2 &
CUDA_VISIBLE_DEVICES=2 python eval_main.py benchmark=$BENCHMARK run=$RUN epoch=5 &
wait $(jobs -p)

CUDA_VISIBLE_DEVICES=0 python eval_main.py benchmark=$BENCHMARK run=$RUN epoch=10 &
CUDA_VISIBLE_DEVICES=1 python eval_main.py benchmark=$BENCHMARK run=$RUN epoch=20 &
CUDA_VISIBLE_DEVICES=2 python eval_main.py benchmark=$BENCHMARK run=$RUN epoch=50 &
wait $(jobs -p)

CUDA_VISIBLE_DEVICES=0 python eval_main.py benchmark=$BENCHMARK run=$RUN epoch=100 &
CUDA_VISIBLE_DEVICES=1 python eval_main.py benchmark=$BENCHMARK run=$RUN epoch=150 &
wait $(jobs -p)
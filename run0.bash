#!/bin/bash

# msp odin mds react dice knn nusa vim ncscore neco epa

trap 'pkill -P $$; exit' SIGINT SIGTERM

BENCHMARK=imagenet
RUN=run0

CUDA_VISIBLE_DEVICES=0 python eval_main.py benchmark=$BENCHMARK run=$RUN pps="[msp,odin,dice,knn,ncscore]" &
CUDA_VISIBLE_DEVICES=1 python eval_main.py benchmark=$BENCHMARK run=$RUN pps="[mds,vim,epa,react,nusa,neco]" &

wait $(jobs -p)

CUDA_VISIBLE_DEVICES=0 python eval_main.py benchmark=$BENCHMARK run=$RUN


#!/bin/bash

trap 'pkill -P $$; exit' SIGINT SIGTERM

BENCHMARK=imagenet200
RUN=run1

CUDA_VISIBLE_DEVICES=0 python eval_main.py benchmark=$BENCHMARK run=$RUN pps="[msp,mds,odin, react]" &
CUDA_VISIBLE_DEVICES=1 python eval_main.py benchmark=$BENCHMARK run=$RUN pps="[dice,knn,vim,epa]" &
CUDA_VISIBLE_DEVICES=2 python eval_main.py benchmark=$BENCHMARK run=$RUN pps="[neco,nusa,ncscore]" &

wait $(jobs -p)

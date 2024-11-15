#!/bin/bash

# msp odin mds react dice knn nusa vim ncscore neco epa

trap 'pkill -P $$; exit' SIGINT SIGTERM

BENCHMARK=imagenet200
RUN=run2

CUDA_VISIBLE_DEVICES=0 python eval_main.py benchmark=$BENCHMARK run=$RUN pps="[msp,odin,dice,knn]" &
CUDA_VISIBLE_DEVICES=1 python eval_main.py benchmark=$BENCHMARK run=$RUN pps="[mds,vim,epa]" &
CUDA_VISIBLE_DEVICES=2 python eval_main.py benchmark=$BENCHMARK run=$RUN pps="[react,nusa,neco,ncscore]" &
wait $(jobs -p)

CUDA_VISIBLE_DEVICES=0 python eval_main.py benchmark=$BENCHMARK run=$RUN
wait $(jobs -p)

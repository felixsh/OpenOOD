#!/bin/bash

# msp odin mds react dice knn nusa vim ncscore neco epa

trap 'pkill -P $$; exit' SIGINT SIGTERM

BENCHMARK=imagenet
RUN=run0

# Already done dice, msp, odin

CUDA_VISIBLE_DEVICES=0 python eval_main.py benchmark=$BENCHMARK run=$RUN pps="[knn]" &
CUDA_VISIBLE_DEVICES=1 python eval_main.py benchmark=$BENCHMARK run=$RUN pps="[mds]" &
CUDA_VISIBLE_DEVICES=2 python eval_main.py benchmark=$BENCHMARK run=$RUN pps="[vim]" &
CUDA_VISIBLE_DEVICES=3 python eval_main.py benchmark=$BENCHMARK run=$RUN pps="[epa]" &
CUDA_VISIBLE_DEVICES=4 python eval_main.py benchmark=$BENCHMARK run=$RUN pps="[react]" &
CUDA_VISIBLE_DEVICES=5 python eval_main.py benchmark=$BENCHMARK run=$RUN pps="[nusa]" &
CUDA_VISIBLE_DEVICES=6 python eval_main.py benchmark=$BENCHMARK run=$RUN pps="[neco]" &
CUDA_VISIBLE_DEVICES=7 python eval_main.py benchmark=$BENCHMARK run=$RUN pps="[ncscore]" &
wait $(jobs -p)

CUDA_VISIBLE_DEVICES=0 python eval_main.py benchmark=$BENCHMARK run=$RUN
wait $(jobs -p)

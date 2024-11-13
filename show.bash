#!/bin/bash

BENCHMARK=imagenet200

python visualize.py benchmark=$BENCHMARK run=run1
#python visualize.py benchmark=$BENCHMARK run=run1
#python visualize.py benchmark=$BENCHMARK run=run2
#python visualize.py benchmark=$BENCHMARK run=run3
#python visualize.py benchmark=$BENCHMARK run=run4
python visualize.py benchmark=$BENCHMARK

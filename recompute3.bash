#!/bin/bash

cleanup() {
  pkill -P $$
  wait
  exit
}

trap cleanup SIGINT SIGTERM

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_50/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_36_34/NCResNet18_32x32_e300_i0.pth method=vim' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_38_55/NCResNet18_32x32_e300_i0.pth method=vim' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_41_40/NCResNet18_32x32_e300_i0.pth method=ncscore' &
wait $(jobs -p)

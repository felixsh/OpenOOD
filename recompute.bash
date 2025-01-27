#!/bin/bash

cleanup() {
  pkill -P $$
  wait
  exit
}

trap cleanup SIGINT SIGTERM

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_31_34/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_31_34/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_31_34/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_31_34/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_31_34/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_31_34/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_31_34/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_31_34/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_31_34/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_31_34/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_31_34/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_31_34/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_31_00/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_31_00/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_31_00/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_31_00/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_31_00/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_31_00/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_31_00/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_31_00/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_31_00/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_31_00/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_31_00/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_31_00/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_28_56/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_28_56/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_28_56/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_28_56/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_28_56/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_28_56/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_28_56/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_28_56/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_28_56/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_28_56/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_28_56/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_28_56/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_26_59/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_26_59/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_26_59/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_26_59/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_26_59/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_26_59/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_26_59/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_26_59/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_26_59/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_26_59/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_26_59/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_26_59/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_26_50/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_26_50/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_26_50/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_26_50/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_26_50/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_26_50/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_26_50/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_26_50/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_26_50/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_26_50/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_26_50/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_26_50/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_00_28/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_00_28/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_00_28/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_00_28/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_00_28/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_00_28/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_00_28/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_00_28/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_00_28/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_00_28/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_00_28/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_00_28/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_59_13/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_59_13/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_59_13/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_59_13/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_59_13/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_59_13/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_59_13/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_59_13/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_59_13/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_59_13/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_59_13/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_59_13/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_41_40/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_41_40/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_41_40/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_41_40/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_41_40/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_41_40/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_41_40/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_41_40/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_41_40/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_41_40/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_41_40/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_41_40/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_41_09/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_41_09/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_41_09/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_41_09/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_41_09/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_41_09/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_41_09/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_41_09/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_41_09/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_41_09/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_41_09/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_41_09/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_38_55/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_38_55/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_38_55/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_38_55/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_38_55/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_38_55/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_38_55/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_38_55/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_38_55/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_38_55/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_38_55/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_38_55/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_36_34/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_36_34/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_36_34/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_36_34/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_36_34/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_36_34/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_36_34/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_36_34/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_36_34/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_36_34/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_36_34/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_36_34/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_35_28/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_35_28/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_35_28/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_35_28/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_35_28/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_35_28/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_35_28/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_35_28/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_35_28/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_35_28/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_35_28/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_35_28/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_10_02/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_10_02/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_10_02/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_10_02/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_10_02/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_10_02/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_10_02/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_10_02/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_10_02/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_10_02/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_10_02/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_10_02/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_09_52/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_09_52/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_09_52/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_09_52/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_09_52/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_09_52/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_09_52/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_09_52/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_09_52/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_09_52/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_09_52/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_09_52/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_51_42/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_51_42/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_51_42/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_51_42/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_51_42/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_51_42/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_51_42/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_51_42/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_51_42/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_51_42/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_51_42/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_51_42/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_51_05/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_51_05/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_51_05/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_51_05/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_51_05/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_51_05/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_51_05/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_51_05/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_51_05/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_51_05/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_51_05/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_51_05/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_48_02/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_48_02/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_48_02/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_48_02/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_48_02/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_48_02/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_48_02/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_48_02/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_48_02/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_48_02/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_48_02/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_48_02/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_46_30/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_46_30/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_46_30/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_46_30/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_46_30/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_46_30/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_46_30/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_46_30/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_46_30/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_46_30/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_46_30/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_46_30/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_45_32/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_45_32/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_45_32/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_45_32/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_45_32/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_45_32/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_45_32/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_45_32/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_45_32/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_45_32/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_45_32/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_45_32/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_17_17/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_17_17/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_17_17/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_17_17/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_17_17/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_17_17/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_17_17/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_17_17/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_17_17/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_17_17/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_17_17/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_17_17/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_17_16/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_17_16/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_17_16/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_17_16/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_17_16/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_17_16/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_17_16/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_17_16/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_17_16/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_17_16/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_17_16/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_17_16/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_00_44/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_00_44/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_00_44/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_00_44/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_00_44/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_00_44/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_00_44/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_00_44/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_00_44/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_00_44/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_00_44/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_00_44/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_00_32/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_00_32/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_00_32/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_00_32/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_00_32/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_00_32/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_00_32/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_00_32/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_00_32/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_00_32/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_00_32/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_00_32/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_57_23/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_57_23/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_57_23/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_57_23/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_57_23/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_57_23/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_57_23/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_57_23/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_57_23/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_57_23/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_57_23/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_57_23/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_56_56/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_56_56/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_56_56/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_56_56/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_56_56/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_56_56/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_56_56/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_56_56/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_56_56/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_56_56/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_56_56/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_56_56/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_56_12/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_56_12/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_56_12/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_56_12/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_56_12/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_56_12/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_56_12/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_56_12/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_56_12/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_56_12/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_56_12/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_56_12/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_08_46/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_08_46/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_08_46/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_08_46/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_08_46/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_08_46/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_08_46/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_08_46/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_08_46/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_08_46/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_08_46/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_08_46/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_08_09/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_08_09/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_08_09/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_08_09/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_08_09/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_08_09/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_08_09/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_08_09/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_08_09/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_08_09/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_08_09/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_08_09/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_07_39/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_07_39/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_07_39/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_07_39/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_07_39/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_07_39/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_07_39/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_07_39/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_07_39/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_07_39/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_07_39/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_07_39/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_07_09/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_07_09/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_07_09/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_07_09/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_07_09/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_07_09/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_07_09/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_07_09/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_07_09/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_07_09/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_07_09/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_07_09/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_06_37/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_06_37/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_06_37/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_06_37/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_06_37/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_06_37/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_06_37/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_06_37/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_06_37/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_06_37/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_06_37/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_06_37/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_06_16/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_06_16/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_06_16/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_06_16/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_06_16/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_06_16/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_06_16/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_06_16/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_06_16/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_06_16/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_06_16/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_06_16/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_53/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_53/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_53/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_53/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_53/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_53/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_53/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_53/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_53/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_53/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_53/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_53/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_27/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_27/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_27/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_27/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_27/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_27/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_27/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_27/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_27/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_27/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_27/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_27/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_00/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_00/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_00/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_00/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_00/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_00/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_00/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_00/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_00/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_00/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_00/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_00/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_04_51/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_04_51/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_04_51/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_04_51/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_04_51/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_04_51/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_04_51/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_04_51/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_04_51/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_04_51/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_04_51/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_04_51/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_50/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_50/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_50/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_50/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_50/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_50/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_50/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_50/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_50/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_50/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_50/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_50/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_22/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_22/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_22/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_22/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_22/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_22/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_22/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_22/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_22/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_22/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_22/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_22/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_04/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_04/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_04/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_04/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_04/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_04/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_04/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_04/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_04/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_04/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_04/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_04/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_01_40/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_01_40/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_01_40/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_01_40/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_01_40/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_01_40/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_01_40/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_01_40/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_01_40/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_01_40/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_01_40/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_01_40/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_01_12/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_01_12/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_01_12/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_01_12/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_01_12/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_01_12/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_01_12/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_01_12/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_01_12/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_01_12/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_01_12/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_01_12/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_00_48/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_00_48/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_00_48/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_00_48/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_00_48/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_00_48/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_00_48/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_00_48/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_00_48/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_00_48/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_00_48/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_00_48/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_06_38/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_06_38/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_06_38/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_06_38/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_06_38/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_06_38/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_06_38/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_06_38/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_06_38/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_06_38/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_06_38/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_06_38/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_06_18/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_06_18/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_06_18/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_06_18/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_06_18/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_06_18/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_06_18/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_06_18/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_06_18/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_06_18/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_06_18/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_06_18/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_05_08/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_05_08/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_05_08/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_05_08/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_05_08/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_05_08/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_05_08/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_05_08/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_05_08/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_05_08/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_05_08/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_05_08/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_59/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_59/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_59/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_59/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_59/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_59/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_59/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_59/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_59/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_59/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_59/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_59/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e300_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e300_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=5 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e300_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=6 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=7 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)


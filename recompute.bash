#!/bin/bash

cleanup() {
  pkill -P $$
  wait
  exit
}

trap cleanup SIGINT SIGTERM

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e1_i0.pth method=nc_train' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e1_i0.pth method=nc_eval' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e1_i0.pth method=msp' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e1_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e1_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e1_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e1_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e1_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e1_i0.pth method=nusa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e1_i0.pth method=vim' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e1_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e1_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e2_i0.pth method=nc_train' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e2_i0.pth method=nc_eval' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e2_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e2_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e2_i0.pth method=mds' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e2_i0.pth method=react' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e2_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e2_i0.pth method=knn' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e2_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e2_i0.pth method=vim' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e2_i0.pth method=ncscore' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e2_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e2_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e5_i0.pth method=nc_train' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e5_i0.pth method=nc_eval' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e5_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e5_i0.pth method=odin' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e5_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e5_i0.pth method=react' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e5_i0.pth method=dice' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e5_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e5_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e5_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e5_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e5_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e5_i0.pth method=epa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e10_i0.pth method=nc_train' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e10_i0.pth method=nc_eval' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e10_i0.pth method=msp' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e10_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e10_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e10_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e10_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e10_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e10_i0.pth method=nusa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e10_i0.pth method=vim' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e10_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e10_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e20_i0.pth method=nc_train' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e20_i0.pth method=nc_eval' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e20_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e20_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e20_i0.pth method=mds' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e20_i0.pth method=react' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e20_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e20_i0.pth method=knn' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e20_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e20_i0.pth method=vim' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e20_i0.pth method=ncscore' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e20_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e20_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e50_i0.pth method=nc_train' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e50_i0.pth method=nc_eval' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e50_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e50_i0.pth method=odin' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e50_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e50_i0.pth method=react' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e50_i0.pth method=dice' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e50_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e50_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e50_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e50_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e50_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e50_i0.pth method=epa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e100_i0.pth method=nc_train' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e100_i0.pth method=nc_eval' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e100_i0.pth method=msp' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e100_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e100_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e100_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e100_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e100_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e100_i0.pth method=nusa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e100_i0.pth method=vim' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e100_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e100_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e200_i0.pth method=nc_train' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e200_i0.pth method=nc_eval' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e200_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e200_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e200_i0.pth method=mds' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e200_i0.pth method=react' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e200_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e200_i0.pth method=knn' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e200_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e200_i0.pth method=vim' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e200_i0.pth method=ncscore' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e200_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e200_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e300_i0.pth method=nc_train' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e300_i0.pth method=nc_eval' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e300_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e300_i0.pth method=odin' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e300_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e300_i0.pth method=react' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e300_i0.pth method=dice' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e300_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e300_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e300_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e300_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e300_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e300_i0.pth method=epa' &
wait $(jobs -p)


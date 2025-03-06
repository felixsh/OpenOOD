#!/bin/bash

cleanup() {
  pkill -P $$
  wait
  exit
}

trap cleanup SIGINT SIGTERM

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_38/NCResNet18_32x32_e1_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_47/NCResNet18_32x32_e1000_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_47/NCResNet18_32x32_e1000_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_47/NCResNet18_32x32_e1000_i0.pth method=neco' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_47/NCResNet18_32x32_e1000_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_47/NCResNet18_32x32_e1000_i0.pth method=acc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e1_i0.pth method=msp' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e1_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e1_i0.pth method=mds' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e1_i0.pth method=react' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e1_i0.pth method=dice' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e1_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e1_i0.pth method=nusa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e1_i0.pth method=vim' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e1_i0.pth method=ncscore' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e1_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e1_i0.pth method=epa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e1_i0.pth method=acc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e2_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e2_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e2_i0.pth method=odin' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e2_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e2_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e2_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e2_i0.pth method=knn' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e2_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e2_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e2_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e2_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e2_i0.pth method=acc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e5_i0.pth method=msp' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e5_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e5_i0.pth method=mds' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e5_i0.pth method=react' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e5_i0.pth method=dice' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e5_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e5_i0.pth method=nusa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e5_i0.pth method=vim' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e5_i0.pth method=ncscore' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e5_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e5_i0.pth method=epa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e5_i0.pth method=acc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e10_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e10_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e10_i0.pth method=odin' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e10_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e10_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e10_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e10_i0.pth method=knn' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e10_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e10_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e10_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e10_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e10_i0.pth method=acc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e20_i0.pth method=msp' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e20_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e20_i0.pth method=mds' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e20_i0.pth method=react' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e20_i0.pth method=dice' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e20_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e20_i0.pth method=nusa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e20_i0.pth method=vim' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e20_i0.pth method=ncscore' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e20_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e20_i0.pth method=epa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e20_i0.pth method=acc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e50_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e50_i0.pth method=odin' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e50_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e50_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e50_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e50_i0.pth method=knn' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e50_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e50_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e50_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e50_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e50_i0.pth method=acc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e100_i0.pth method=msp' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e100_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e100_i0.pth method=mds' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e100_i0.pth method=react' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e100_i0.pth method=dice' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e100_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e100_i0.pth method=nusa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e100_i0.pth method=vim' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e100_i0.pth method=ncscore' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e100_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e100_i0.pth method=epa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e100_i0.pth method=acc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e200_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e200_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e200_i0.pth method=odin' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e200_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e200_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e200_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e200_i0.pth method=knn' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e200_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e200_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e200_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e200_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e200_i0.pth method=acc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e500_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e500_i0.pth method=msp' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e500_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e500_i0.pth method=mds' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e500_i0.pth method=react' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e500_i0.pth method=dice' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e500_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e500_i0.pth method=nusa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e500_i0.pth method=vim' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e500_i0.pth method=ncscore' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e500_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e500_i0.pth method=epa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e500_i0.pth method=acc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e1000_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e1000_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e1000_i0.pth method=odin' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e1000_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e1000_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e1000_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e1000_i0.pth method=knn' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e1000_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e1000_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e1000_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e1000_i0.pth method=neco' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e1000_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_51/NCResNet18_32x32_e1000_i0.pth method=acc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e1_i0.pth method=msp' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e1_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e1_i0.pth method=mds' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e1_i0.pth method=react' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e1_i0.pth method=dice' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e1_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e1_i0.pth method=nusa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e1_i0.pth method=vim' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e1_i0.pth method=ncscore' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e1_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e1_i0.pth method=epa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e1_i0.pth method=acc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e2_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e2_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e2_i0.pth method=odin' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e2_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e2_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e2_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e2_i0.pth method=knn' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e2_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e2_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e2_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e2_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e2_i0.pth method=acc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e5_i0.pth method=msp' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e5_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e5_i0.pth method=mds' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e5_i0.pth method=react' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e5_i0.pth method=dice' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e5_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e5_i0.pth method=nusa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e5_i0.pth method=vim' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e5_i0.pth method=ncscore' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e5_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e5_i0.pth method=epa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e5_i0.pth method=acc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e10_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e10_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e10_i0.pth method=odin' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e10_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e10_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e10_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e10_i0.pth method=knn' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e10_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e10_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e10_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e10_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e10_i0.pth method=acc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e20_i0.pth method=msp' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e20_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e20_i0.pth method=mds' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e20_i0.pth method=react' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e20_i0.pth method=dice' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e20_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e20_i0.pth method=nusa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e20_i0.pth method=vim' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e20_i0.pth method=ncscore' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e20_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e20_i0.pth method=epa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e20_i0.pth method=acc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e50_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e50_i0.pth method=odin' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e50_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e50_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e50_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e50_i0.pth method=knn' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e50_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e50_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e50_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e50_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e50_i0.pth method=acc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e100_i0.pth method=msp' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e100_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e100_i0.pth method=mds' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e100_i0.pth method=react' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e100_i0.pth method=dice' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e100_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e100_i0.pth method=nusa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e100_i0.pth method=vim' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e100_i0.pth method=ncscore' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e100_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e100_i0.pth method=epa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e100_i0.pth method=acc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e200_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e200_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e200_i0.pth method=odin' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e200_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e200_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e200_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e200_i0.pth method=knn' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e200_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e200_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e200_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e200_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e200_i0.pth method=acc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e500_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e500_i0.pth method=msp' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e500_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e500_i0.pth method=mds' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e500_i0.pth method=react' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e500_i0.pth method=dice' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e500_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e500_i0.pth method=nusa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e500_i0.pth method=vim' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e500_i0.pth method=ncscore' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e500_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e500_i0.pth method=epa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e500_i0.pth method=acc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e1000_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e1000_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e1000_i0.pth method=odin' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e1000_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e1000_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e1000_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e1000_i0.pth method=knn' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e1000_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e1000_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e1000_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e1000_i0.pth method=neco' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e1000_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/type/no_noise/1000+_epochs/cifar100_1000_run_e1000_2025_03_05-02_06_54/NCResNet18_32x32_e1000_i0.pth method=acc' &
wait $(jobs -p)


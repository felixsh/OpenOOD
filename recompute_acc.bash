#!/bin/bash

cleanup() {
  pkill -P $$
  wait
  exit
}

trap cleanup SIGINT SIGTERM

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e1_i0.pth i=1 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e2_i0.pth i=2 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e5_i0.pth i=3 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e10_i0.pth i=4 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e20_i0.pth i=5 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e50_i0.pth i=6 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e100_i0.pth i=7 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e200_i0.pth i=8 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e300_i0.pth i=9 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e1_i0.pth i=10 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e2_i0.pth i=11 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e5_i0.pth i=12 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e10_i0.pth i=13 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e20_i0.pth i=14 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e50_i0.pth i=15 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e100_i0.pth i=16 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e200_i0.pth i=17 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e300_i0.pth i=18 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e1_i0.pth i=19 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e2_i0.pth i=20 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e5_i0.pth i=21 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e10_i0.pth i=22 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e20_i0.pth i=23 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e50_i0.pth i=24 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e100_i0.pth i=25 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e200_i0.pth i=26 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e300_i0.pth i=27 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e1_i0.pth i=28 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e2_i0.pth i=29 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e5_i0.pth i=30 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e10_i0.pth i=31 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e20_i0.pth i=32 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e50_i0.pth i=33 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e100_i0.pth i=34 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e200_i0.pth i=35 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e300_i0.pth i=36 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e1_i0.pth i=37 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e2_i0.pth i=38 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e5_i0.pth i=39 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e10_i0.pth i=40 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e20_i0.pth i=41 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e50_i0.pth i=42 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e100_i0.pth i=43 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e200_i0.pth i=44 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e300_i0.pth i=45 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e1_i0.pth i=46 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e2_i0.pth i=47 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e5_i0.pth i=48 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e10_i0.pth i=49 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e20_i0.pth i=50 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e50_i0.pth i=51 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e100_i0.pth i=52 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e200_i0.pth i=53 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e300_i0.pth i=54 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e1_i0.pth i=55 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e2_i0.pth i=56 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e5_i0.pth i=57 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e10_i0.pth i=58 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e20_i0.pth i=59 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e50_i0.pth i=60 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e100_i0.pth i=61 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e200_i0.pth i=62 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e300_i0.pth i=63 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e1_i0.pth i=64 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e2_i0.pth i=65 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e5_i0.pth i=66 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e10_i0.pth i=67 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e20_i0.pth i=68 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e50_i0.pth i=69 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e100_i0.pth i=70 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e200_i0.pth i=71 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e300_i0.pth i=72 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e1_i0.pth i=73 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e2_i0.pth i=74 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e5_i0.pth i=75 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e10_i0.pth i=76 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e20_i0.pth i=77 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e50_i0.pth i=78 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e100_i0.pth i=79 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e200_i0.pth i=80 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e300_i0.pth i=81 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_59/NCResNet18_32x32_e1_i0.pth i=82 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_59/NCResNet18_32x32_e2_i0.pth i=83 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_59/NCResNet18_32x32_e5_i0.pth i=84 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_59/NCResNet18_32x32_e10_i0.pth i=85 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_59/NCResNet18_32x32_e20_i0.pth i=86 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_59/NCResNet18_32x32_e50_i0.pth i=87 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_59/NCResNet18_32x32_e100_i0.pth i=88 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_59/NCResNet18_32x32_e200_i0.pth i=89 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_59/NCResNet18_32x32_e300_i0.pth i=90 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_05_08/NCResNet18_32x32_e1_i0.pth i=91 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_05_08/NCResNet18_32x32_e2_i0.pth i=92 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_05_08/NCResNet18_32x32_e5_i0.pth i=93 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_05_08/NCResNet18_32x32_e10_i0.pth i=94 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_05_08/NCResNet18_32x32_e20_i0.pth i=95 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_05_08/NCResNet18_32x32_e50_i0.pth i=96 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_05_08/NCResNet18_32x32_e100_i0.pth i=97 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_05_08/NCResNet18_32x32_e200_i0.pth i=98 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_05_08/NCResNet18_32x32_e300_i0.pth i=99 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_06_18/NCResNet18_32x32_e1_i0.pth i=100 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_06_18/NCResNet18_32x32_e2_i0.pth i=101 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_06_18/NCResNet18_32x32_e5_i0.pth i=102 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_06_18/NCResNet18_32x32_e10_i0.pth i=103 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_06_18/NCResNet18_32x32_e20_i0.pth i=104 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_06_18/NCResNet18_32x32_e50_i0.pth i=105 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_06_18/NCResNet18_32x32_e100_i0.pth i=106 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_06_18/NCResNet18_32x32_e200_i0.pth i=107 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_06_18/NCResNet18_32x32_e300_i0.pth i=108 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_06_38/NCResNet18_32x32_e1_i0.pth i=109 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_06_38/NCResNet18_32x32_e2_i0.pth i=110 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_06_38/NCResNet18_32x32_e5_i0.pth i=111 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_06_38/NCResNet18_32x32_e10_i0.pth i=112 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_06_38/NCResNet18_32x32_e20_i0.pth i=113 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_06_38/NCResNet18_32x32_e50_i0.pth i=114 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_06_38/NCResNet18_32x32_e100_i0.pth i=115 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_06_38/NCResNet18_32x32_e200_i0.pth i=116 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_06_38/NCResNet18_32x32_e300_i0.pth i=117 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_00_48/NCResNet18_32x32_e1_i0.pth i=118 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_00_48/NCResNet18_32x32_e2_i0.pth i=119 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_00_48/NCResNet18_32x32_e5_i0.pth i=120 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_00_48/NCResNet18_32x32_e10_i0.pth i=121 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_00_48/NCResNet18_32x32_e20_i0.pth i=122 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_00_48/NCResNet18_32x32_e50_i0.pth i=123 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_00_48/NCResNet18_32x32_e100_i0.pth i=124 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_00_48/NCResNet18_32x32_e200_i0.pth i=125 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_00_48/NCResNet18_32x32_e300_i0.pth i=126 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_01_12/NCResNet18_32x32_e1_i0.pth i=127 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_01_12/NCResNet18_32x32_e2_i0.pth i=128 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_01_12/NCResNet18_32x32_e5_i0.pth i=129 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_01_12/NCResNet18_32x32_e10_i0.pth i=130 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_01_12/NCResNet18_32x32_e20_i0.pth i=131 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_01_12/NCResNet18_32x32_e50_i0.pth i=132 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_01_12/NCResNet18_32x32_e100_i0.pth i=133 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_01_12/NCResNet18_32x32_e200_i0.pth i=134 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_01_12/NCResNet18_32x32_e300_i0.pth i=135 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_01_40/NCResNet18_32x32_e1_i0.pth i=136 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_01_40/NCResNet18_32x32_e2_i0.pth i=137 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_01_40/NCResNet18_32x32_e5_i0.pth i=138 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_01_40/NCResNet18_32x32_e10_i0.pth i=139 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_01_40/NCResNet18_32x32_e20_i0.pth i=140 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_01_40/NCResNet18_32x32_e50_i0.pth i=141 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_01_40/NCResNet18_32x32_e100_i0.pth i=142 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_01_40/NCResNet18_32x32_e200_i0.pth i=143 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_01_40/NCResNet18_32x32_e300_i0.pth i=144 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_04/NCResNet18_32x32_e1_i0.pth i=145 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_04/NCResNet18_32x32_e2_i0.pth i=146 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_04/NCResNet18_32x32_e5_i0.pth i=147 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_04/NCResNet18_32x32_e10_i0.pth i=148 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_04/NCResNet18_32x32_e20_i0.pth i=149 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_04/NCResNet18_32x32_e50_i0.pth i=150 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_04/NCResNet18_32x32_e100_i0.pth i=151 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_04/NCResNet18_32x32_e200_i0.pth i=152 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_04/NCResNet18_32x32_e300_i0.pth i=153 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_22/NCResNet18_32x32_e1_i0.pth i=154 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_22/NCResNet18_32x32_e2_i0.pth i=155 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_22/NCResNet18_32x32_e5_i0.pth i=156 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_22/NCResNet18_32x32_e10_i0.pth i=157 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_22/NCResNet18_32x32_e20_i0.pth i=158 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_22/NCResNet18_32x32_e50_i0.pth i=159 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_22/NCResNet18_32x32_e100_i0.pth i=160 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_22/NCResNet18_32x32_e200_i0.pth i=161 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_22/NCResNet18_32x32_e300_i0.pth i=162 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_50/NCResNet18_32x32_e1_i0.pth i=163 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_50/NCResNet18_32x32_e2_i0.pth i=164 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_50/NCResNet18_32x32_e5_i0.pth i=165 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_50/NCResNet18_32x32_e10_i0.pth i=166 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_50/NCResNet18_32x32_e20_i0.pth i=167 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_50/NCResNet18_32x32_e50_i0.pth i=168 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_50/NCResNet18_32x32_e100_i0.pth i=169 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_50/NCResNet18_32x32_e200_i0.pth i=170 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_50/NCResNet18_32x32_e300_i0.pth i=171 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_04_51/NCResNet18_32x32_e1_i0.pth i=172 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_04_51/NCResNet18_32x32_e2_i0.pth i=173 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_04_51/NCResNet18_32x32_e5_i0.pth i=174 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_04_51/NCResNet18_32x32_e10_i0.pth i=175 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_04_51/NCResNet18_32x32_e20_i0.pth i=176 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_04_51/NCResNet18_32x32_e50_i0.pth i=177 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_04_51/NCResNet18_32x32_e100_i0.pth i=178 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_04_51/NCResNet18_32x32_e200_i0.pth i=179 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_04_51/NCResNet18_32x32_e300_i0.pth i=180 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_00/NCResNet18_32x32_e1_i0.pth i=181 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_00/NCResNet18_32x32_e2_i0.pth i=182 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_00/NCResNet18_32x32_e5_i0.pth i=183 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_00/NCResNet18_32x32_e10_i0.pth i=184 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_00/NCResNet18_32x32_e20_i0.pth i=185 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_00/NCResNet18_32x32_e50_i0.pth i=186 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_00/NCResNet18_32x32_e100_i0.pth i=187 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_00/NCResNet18_32x32_e200_i0.pth i=188 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_00/NCResNet18_32x32_e300_i0.pth i=189 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_27/NCResNet18_32x32_e1_i0.pth i=190 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_27/NCResNet18_32x32_e2_i0.pth i=191 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_27/NCResNet18_32x32_e5_i0.pth i=192 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_27/NCResNet18_32x32_e10_i0.pth i=193 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_27/NCResNet18_32x32_e20_i0.pth i=194 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_27/NCResNet18_32x32_e50_i0.pth i=195 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_27/NCResNet18_32x32_e100_i0.pth i=196 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_27/NCResNet18_32x32_e200_i0.pth i=197 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_27/NCResNet18_32x32_e300_i0.pth i=198 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_53/NCResNet18_32x32_e1_i0.pth i=199 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_53/NCResNet18_32x32_e2_i0.pth i=200 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_53/NCResNet18_32x32_e5_i0.pth i=201 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_53/NCResNet18_32x32_e10_i0.pth i=202 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_53/NCResNet18_32x32_e20_i0.pth i=203 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_53/NCResNet18_32x32_e50_i0.pth i=204 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_53/NCResNet18_32x32_e100_i0.pth i=205 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_53/NCResNet18_32x32_e200_i0.pth i=206 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_05_53/NCResNet18_32x32_e300_i0.pth i=207 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_06_16/NCResNet18_32x32_e1_i0.pth i=208 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_06_16/NCResNet18_32x32_e2_i0.pth i=209 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_06_16/NCResNet18_32x32_e5_i0.pth i=210 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_06_16/NCResNet18_32x32_e10_i0.pth i=211 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_06_16/NCResNet18_32x32_e20_i0.pth i=212 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_06_16/NCResNet18_32x32_e50_i0.pth i=213 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_06_16/NCResNet18_32x32_e100_i0.pth i=214 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_06_16/NCResNet18_32x32_e200_i0.pth i=215 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_06_16/NCResNet18_32x32_e300_i0.pth i=216 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_06_37/NCResNet18_32x32_e1_i0.pth i=217 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_06_37/NCResNet18_32x32_e2_i0.pth i=218 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_06_37/NCResNet18_32x32_e5_i0.pth i=219 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_06_37/NCResNet18_32x32_e10_i0.pth i=220 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_06_37/NCResNet18_32x32_e20_i0.pth i=221 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_06_37/NCResNet18_32x32_e50_i0.pth i=222 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_06_37/NCResNet18_32x32_e100_i0.pth i=223 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_06_37/NCResNet18_32x32_e200_i0.pth i=224 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_06_37/NCResNet18_32x32_e300_i0.pth i=225 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_07_09/NCResNet18_32x32_e1_i0.pth i=226 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_07_09/NCResNet18_32x32_e2_i0.pth i=227 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_07_09/NCResNet18_32x32_e5_i0.pth i=228 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_07_09/NCResNet18_32x32_e10_i0.pth i=229 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_07_09/NCResNet18_32x32_e20_i0.pth i=230 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_07_09/NCResNet18_32x32_e50_i0.pth i=231 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_07_09/NCResNet18_32x32_e100_i0.pth i=232 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_07_09/NCResNet18_32x32_e200_i0.pth i=233 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_07_09/NCResNet18_32x32_e300_i0.pth i=234 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_07_39/NCResNet18_32x32_e1_i0.pth i=235 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_07_39/NCResNet18_32x32_e2_i0.pth i=236 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_07_39/NCResNet18_32x32_e5_i0.pth i=237 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_07_39/NCResNet18_32x32_e10_i0.pth i=238 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_07_39/NCResNet18_32x32_e20_i0.pth i=239 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_07_39/NCResNet18_32x32_e50_i0.pth i=240 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_07_39/NCResNet18_32x32_e100_i0.pth i=241 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_07_39/NCResNet18_32x32_e200_i0.pth i=242 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_07_39/NCResNet18_32x32_e300_i0.pth i=243 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_08_09/NCResNet18_32x32_e1_i0.pth i=244 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_08_09/NCResNet18_32x32_e2_i0.pth i=245 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_08_09/NCResNet18_32x32_e5_i0.pth i=246 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_08_09/NCResNet18_32x32_e10_i0.pth i=247 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_08_09/NCResNet18_32x32_e20_i0.pth i=248 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_08_09/NCResNet18_32x32_e50_i0.pth i=249 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_08_09/NCResNet18_32x32_e100_i0.pth i=250 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_08_09/NCResNet18_32x32_e200_i0.pth i=251 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_08_09/NCResNet18_32x32_e300_i0.pth i=252 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_08_46/NCResNet18_32x32_e1_i0.pth i=253 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_08_46/NCResNet18_32x32_e2_i0.pth i=254 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_08_46/NCResNet18_32x32_e5_i0.pth i=255 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_08_46/NCResNet18_32x32_e10_i0.pth i=256 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_08_46/NCResNet18_32x32_e20_i0.pth i=257 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_08_46/NCResNet18_32x32_e50_i0.pth i=258 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_08_46/NCResNet18_32x32_e100_i0.pth i=259 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_08_46/NCResNet18_32x32_e200_i0.pth i=260 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_08_46/NCResNet18_32x32_e300_i0.pth i=261 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_56_12/NCResNet18_32x32_e1_i0.pth i=262 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_56_12/NCResNet18_32x32_e2_i0.pth i=263 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_56_12/NCResNet18_32x32_e5_i0.pth i=264 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_56_12/NCResNet18_32x32_e10_i0.pth i=265 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_56_12/NCResNet18_32x32_e20_i0.pth i=266 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_56_12/NCResNet18_32x32_e50_i0.pth i=267 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_56_12/NCResNet18_32x32_e100_i0.pth i=268 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_56_12/NCResNet18_32x32_e200_i0.pth i=269 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_56_12/NCResNet18_32x32_e300_i0.pth i=270 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_56_56/NCResNet18_32x32_e1_i0.pth i=271 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_56_56/NCResNet18_32x32_e2_i0.pth i=272 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_56_56/NCResNet18_32x32_e5_i0.pth i=273 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_56_56/NCResNet18_32x32_e10_i0.pth i=274 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_56_56/NCResNet18_32x32_e20_i0.pth i=275 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_56_56/NCResNet18_32x32_e50_i0.pth i=276 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_56_56/NCResNet18_32x32_e100_i0.pth i=277 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_56_56/NCResNet18_32x32_e200_i0.pth i=278 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_56_56/NCResNet18_32x32_e300_i0.pth i=279 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_57_23/NCResNet18_32x32_e1_i0.pth i=280 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_57_23/NCResNet18_32x32_e2_i0.pth i=281 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_57_23/NCResNet18_32x32_e5_i0.pth i=282 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_57_23/NCResNet18_32x32_e10_i0.pth i=283 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_57_23/NCResNet18_32x32_e20_i0.pth i=284 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_57_23/NCResNet18_32x32_e50_i0.pth i=285 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_57_23/NCResNet18_32x32_e100_i0.pth i=286 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_57_23/NCResNet18_32x32_e200_i0.pth i=287 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_57_23/NCResNet18_32x32_e300_i0.pth i=288 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_00_32/NCResNet18_32x32_e1_i0.pth i=289 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_00_32/NCResNet18_32x32_e2_i0.pth i=290 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_00_32/NCResNet18_32x32_e5_i0.pth i=291 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_00_32/NCResNet18_32x32_e10_i0.pth i=292 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_00_32/NCResNet18_32x32_e20_i0.pth i=293 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_00_32/NCResNet18_32x32_e50_i0.pth i=294 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_00_32/NCResNet18_32x32_e100_i0.pth i=295 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_00_32/NCResNet18_32x32_e200_i0.pth i=296 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_00_32/NCResNet18_32x32_e300_i0.pth i=297 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_00_44/NCResNet18_32x32_e1_i0.pth i=298 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_00_44/NCResNet18_32x32_e2_i0.pth i=299 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_00_44/NCResNet18_32x32_e5_i0.pth i=300 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_00_44/NCResNet18_32x32_e10_i0.pth i=301 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_00_44/NCResNet18_32x32_e20_i0.pth i=302 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_00_44/NCResNet18_32x32_e50_i0.pth i=303 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_00_44/NCResNet18_32x32_e100_i0.pth i=304 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_00_44/NCResNet18_32x32_e200_i0.pth i=305 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_00_44/NCResNet18_32x32_e300_i0.pth i=306 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_17_16/NCResNet18_32x32_e1_i0.pth i=307 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_17_16/NCResNet18_32x32_e2_i0.pth i=308 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_17_16/NCResNet18_32x32_e5_i0.pth i=309 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_17_16/NCResNet18_32x32_e10_i0.pth i=310 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_17_16/NCResNet18_32x32_e20_i0.pth i=311 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_17_16/NCResNet18_32x32_e50_i0.pth i=312 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_17_16/NCResNet18_32x32_e100_i0.pth i=313 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_17_16/NCResNet18_32x32_e200_i0.pth i=314 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_17_16/NCResNet18_32x32_e300_i0.pth i=315 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_17_17/NCResNet18_32x32_e1_i0.pth i=316 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_17_17/NCResNet18_32x32_e2_i0.pth i=317 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_17_17/NCResNet18_32x32_e5_i0.pth i=318 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_17_17/NCResNet18_32x32_e10_i0.pth i=319 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_17_17/NCResNet18_32x32_e20_i0.pth i=320 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_17_17/NCResNet18_32x32_e50_i0.pth i=321 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_17_17/NCResNet18_32x32_e100_i0.pth i=322 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_17_17/NCResNet18_32x32_e200_i0.pth i=323 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_17_17/NCResNet18_32x32_e300_i0.pth i=324 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_45_32/NCResNet18_32x32_e1_i0.pth i=325 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_45_32/NCResNet18_32x32_e2_i0.pth i=326 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_45_32/NCResNet18_32x32_e5_i0.pth i=327 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_45_32/NCResNet18_32x32_e10_i0.pth i=328 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_45_32/NCResNet18_32x32_e20_i0.pth i=329 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_45_32/NCResNet18_32x32_e50_i0.pth i=330 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_45_32/NCResNet18_32x32_e100_i0.pth i=331 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_45_32/NCResNet18_32x32_e200_i0.pth i=332 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_45_32/NCResNet18_32x32_e300_i0.pth i=333 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_46_30/NCResNet18_32x32_e1_i0.pth i=334 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_46_30/NCResNet18_32x32_e2_i0.pth i=335 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_46_30/NCResNet18_32x32_e5_i0.pth i=336 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_46_30/NCResNet18_32x32_e10_i0.pth i=337 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_46_30/NCResNet18_32x32_e20_i0.pth i=338 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_46_30/NCResNet18_32x32_e50_i0.pth i=339 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_46_30/NCResNet18_32x32_e100_i0.pth i=340 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_46_30/NCResNet18_32x32_e200_i0.pth i=341 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_46_30/NCResNet18_32x32_e300_i0.pth i=342 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_48_02/NCResNet18_32x32_e1_i0.pth i=343 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_48_02/NCResNet18_32x32_e2_i0.pth i=344 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_48_02/NCResNet18_32x32_e5_i0.pth i=345 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_48_02/NCResNet18_32x32_e10_i0.pth i=346 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_48_02/NCResNet18_32x32_e20_i0.pth i=347 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_48_02/NCResNet18_32x32_e50_i0.pth i=348 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_48_02/NCResNet18_32x32_e100_i0.pth i=349 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_48_02/NCResNet18_32x32_e200_i0.pth i=350 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_48_02/NCResNet18_32x32_e300_i0.pth i=351 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_51_05/NCResNet18_32x32_e1_i0.pth i=352 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_51_05/NCResNet18_32x32_e2_i0.pth i=353 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_51_05/NCResNet18_32x32_e5_i0.pth i=354 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_51_05/NCResNet18_32x32_e10_i0.pth i=355 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_51_05/NCResNet18_32x32_e20_i0.pth i=356 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_51_05/NCResNet18_32x32_e50_i0.pth i=357 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_51_05/NCResNet18_32x32_e100_i0.pth i=358 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_51_05/NCResNet18_32x32_e200_i0.pth i=359 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_51_05/NCResNet18_32x32_e300_i0.pth i=360 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_51_42/NCResNet18_32x32_e1_i0.pth i=361 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_51_42/NCResNet18_32x32_e2_i0.pth i=362 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_51_42/NCResNet18_32x32_e5_i0.pth i=363 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_51_42/NCResNet18_32x32_e10_i0.pth i=364 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_51_42/NCResNet18_32x32_e20_i0.pth i=365 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_51_42/NCResNet18_32x32_e50_i0.pth i=366 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_51_42/NCResNet18_32x32_e100_i0.pth i=367 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_51_42/NCResNet18_32x32_e200_i0.pth i=368 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-04_51_42/NCResNet18_32x32_e300_i0.pth i=369 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_09_52/NCResNet18_32x32_e1_i0.pth i=370 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_09_52/NCResNet18_32x32_e2_i0.pth i=371 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_09_52/NCResNet18_32x32_e5_i0.pth i=372 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_09_52/NCResNet18_32x32_e10_i0.pth i=373 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_09_52/NCResNet18_32x32_e20_i0.pth i=374 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_09_52/NCResNet18_32x32_e50_i0.pth i=375 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_09_52/NCResNet18_32x32_e100_i0.pth i=376 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_09_52/NCResNet18_32x32_e200_i0.pth i=377 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_09_52/NCResNet18_32x32_e300_i0.pth i=378 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_10_02/NCResNet18_32x32_e1_i0.pth i=379 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_10_02/NCResNet18_32x32_e2_i0.pth i=380 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_10_02/NCResNet18_32x32_e5_i0.pth i=381 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_10_02/NCResNet18_32x32_e10_i0.pth i=382 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_10_02/NCResNet18_32x32_e20_i0.pth i=383 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_10_02/NCResNet18_32x32_e50_i0.pth i=384 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_10_02/NCResNet18_32x32_e100_i0.pth i=385 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_10_02/NCResNet18_32x32_e200_i0.pth i=386 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_10_02/NCResNet18_32x32_e300_i0.pth i=387 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_35_28/NCResNet18_32x32_e1_i0.pth i=388 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_35_28/NCResNet18_32x32_e2_i0.pth i=389 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_35_28/NCResNet18_32x32_e5_i0.pth i=390 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_35_28/NCResNet18_32x32_e10_i0.pth i=391 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_35_28/NCResNet18_32x32_e20_i0.pth i=392 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_35_28/NCResNet18_32x32_e50_i0.pth i=393 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_35_28/NCResNet18_32x32_e100_i0.pth i=394 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_35_28/NCResNet18_32x32_e200_i0.pth i=395 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_35_28/NCResNet18_32x32_e300_i0.pth i=396 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_36_34/NCResNet18_32x32_e1_i0.pth i=397 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_36_34/NCResNet18_32x32_e2_i0.pth i=398 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_36_34/NCResNet18_32x32_e5_i0.pth i=399 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_36_34/NCResNet18_32x32_e10_i0.pth i=400 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_36_34/NCResNet18_32x32_e20_i0.pth i=401 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_36_34/NCResNet18_32x32_e50_i0.pth i=402 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_36_34/NCResNet18_32x32_e100_i0.pth i=403 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_36_34/NCResNet18_32x32_e200_i0.pth i=404 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_36_34/NCResNet18_32x32_e300_i0.pth i=405 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_38_55/NCResNet18_32x32_e1_i0.pth i=406 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_38_55/NCResNet18_32x32_e2_i0.pth i=407 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_38_55/NCResNet18_32x32_e5_i0.pth i=408 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_38_55/NCResNet18_32x32_e10_i0.pth i=409 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_38_55/NCResNet18_32x32_e20_i0.pth i=410 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_38_55/NCResNet18_32x32_e50_i0.pth i=411 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_38_55/NCResNet18_32x32_e100_i0.pth i=412 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_38_55/NCResNet18_32x32_e200_i0.pth i=413 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_38_55/NCResNet18_32x32_e300_i0.pth i=414 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_41_09/NCResNet18_32x32_e1_i0.pth i=415 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_41_09/NCResNet18_32x32_e2_i0.pth i=416 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_41_09/NCResNet18_32x32_e5_i0.pth i=417 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_41_09/NCResNet18_32x32_e10_i0.pth i=418 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_41_09/NCResNet18_32x32_e20_i0.pth i=419 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_41_09/NCResNet18_32x32_e50_i0.pth i=420 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_41_09/NCResNet18_32x32_e100_i0.pth i=421 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_41_09/NCResNet18_32x32_e200_i0.pth i=422 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_41_09/NCResNet18_32x32_e300_i0.pth i=423 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_41_40/NCResNet18_32x32_e1_i0.pth i=424 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_41_40/NCResNet18_32x32_e2_i0.pth i=425 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_41_40/NCResNet18_32x32_e5_i0.pth i=426 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_41_40/NCResNet18_32x32_e10_i0.pth i=427 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_41_40/NCResNet18_32x32_e20_i0.pth i=428 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_41_40/NCResNet18_32x32_e50_i0.pth i=429 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_41_40/NCResNet18_32x32_e100_i0.pth i=430 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_41_40/NCResNet18_32x32_e200_i0.pth i=431 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_41_40/NCResNet18_32x32_e300_i0.pth i=432 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_59_13/NCResNet18_32x32_e1_i0.pth i=433 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_59_13/NCResNet18_32x32_e2_i0.pth i=434 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_59_13/NCResNet18_32x32_e5_i0.pth i=435 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_59_13/NCResNet18_32x32_e10_i0.pth i=436 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_59_13/NCResNet18_32x32_e20_i0.pth i=437 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_59_13/NCResNet18_32x32_e50_i0.pth i=438 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_59_13/NCResNet18_32x32_e100_i0.pth i=439 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_59_13/NCResNet18_32x32_e200_i0.pth i=440 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-05_59_13/NCResNet18_32x32_e300_i0.pth i=441 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_00_28/NCResNet18_32x32_e1_i0.pth i=442 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_00_28/NCResNet18_32x32_e2_i0.pth i=443 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_00_28/NCResNet18_32x32_e5_i0.pth i=444 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_00_28/NCResNet18_32x32_e10_i0.pth i=445 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_00_28/NCResNet18_32x32_e20_i0.pth i=446 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_00_28/NCResNet18_32x32_e50_i0.pth i=447 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_00_28/NCResNet18_32x32_e100_i0.pth i=448 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_00_28/NCResNet18_32x32_e200_i0.pth i=449 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_00_28/NCResNet18_32x32_e300_i0.pth i=450 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_26_50/NCResNet18_32x32_e1_i0.pth i=451 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_26_50/NCResNet18_32x32_e2_i0.pth i=452 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_26_50/NCResNet18_32x32_e5_i0.pth i=453 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_26_50/NCResNet18_32x32_e10_i0.pth i=454 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_26_50/NCResNet18_32x32_e20_i0.pth i=455 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_26_50/NCResNet18_32x32_e50_i0.pth i=456 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_26_50/NCResNet18_32x32_e100_i0.pth i=457 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_26_50/NCResNet18_32x32_e200_i0.pth i=458 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_26_50/NCResNet18_32x32_e300_i0.pth i=459 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_26_59/NCResNet18_32x32_e1_i0.pth i=460 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_26_59/NCResNet18_32x32_e2_i0.pth i=461 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_26_59/NCResNet18_32x32_e5_i0.pth i=462 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_26_59/NCResNet18_32x32_e10_i0.pth i=463 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_26_59/NCResNet18_32x32_e20_i0.pth i=464 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_26_59/NCResNet18_32x32_e50_i0.pth i=465 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_26_59/NCResNet18_32x32_e100_i0.pth i=466 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_26_59/NCResNet18_32x32_e200_i0.pth i=467 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_26_59/NCResNet18_32x32_e300_i0.pth i=468 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_28_56/NCResNet18_32x32_e1_i0.pth i=469 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_28_56/NCResNet18_32x32_e2_i0.pth i=470 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_28_56/NCResNet18_32x32_e5_i0.pth i=471 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_28_56/NCResNet18_32x32_e10_i0.pth i=472 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_28_56/NCResNet18_32x32_e20_i0.pth i=473 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_28_56/NCResNet18_32x32_e50_i0.pth i=474 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_28_56/NCResNet18_32x32_e100_i0.pth i=475 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_28_56/NCResNet18_32x32_e200_i0.pth i=476 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_28_56/NCResNet18_32x32_e300_i0.pth i=477 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_31_00/NCResNet18_32x32_e1_i0.pth i=478 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_31_00/NCResNet18_32x32_e2_i0.pth i=479 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_31_00/NCResNet18_32x32_e5_i0.pth i=480 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_31_00/NCResNet18_32x32_e10_i0.pth i=481 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_31_00/NCResNet18_32x32_e20_i0.pth i=482 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_31_00/NCResNet18_32x32_e50_i0.pth i=483 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_31_00/NCResNet18_32x32_e100_i0.pth i=484 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_31_00/NCResNet18_32x32_e200_i0.pth i=485 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_31_00/NCResNet18_32x32_e300_i0.pth i=486 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_31_34/NCResNet18_32x32_e1_i0.pth i=487 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_31_34/NCResNet18_32x32_e2_i0.pth i=488 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_31_34/NCResNet18_32x32_e5_i0.pth i=489 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_31_34/NCResNet18_32x32_e10_i0.pth i=490 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_31_34/NCResNet18_32x32_e20_i0.pth i=491 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_31_34/NCResNet18_32x32_e50_i0.pth i=492 n=495' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_31_34/NCResNet18_32x32_e100_i0.pth i=493 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_31_34/NCResNet18_32x32_e200_i0.pth i=494 n=495' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-06_31_34/NCResNet18_32x32_e300_i0.pth i=495 n=495' &
wait $(jobs -p)


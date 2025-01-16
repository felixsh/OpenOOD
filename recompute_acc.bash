#!/bin/bash

cleanup() {
  pkill -P $$
  wait
  exit
}

trap cleanup SIGINT SIGTERM

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet/ResNet50/no_noise/150+_epochs/run_e150_2024_11_12-21_40_57/_FabricModule_e150_i0.pth i=1 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet/ResNet50/no_noise/150+_epochs/run_e150_2024_11_12-21_40_57/_FabricModule_e100_i0.pth i=2 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet/ResNet50/no_noise/150+_epochs/run_e150_2024_11_12-21_40_57/_FabricModule_e50_i0.pth i=3 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet/ResNet50/no_noise/150+_epochs/run_e150_2024_11_12-21_40_57/_FabricModule_e20_i0.pth i=4 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet/ResNet50/no_noise/150+_epochs/run_e150_2024_11_12-21_40_57/_FabricModule_e10_i0.pth i=5 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet/ResNet50/no_noise/150+_epochs/run_e150_2024_11_12-21_40_57/_FabricModule_e5_i0.pth i=6 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet/ResNet50/no_noise/150+_epochs/run_e150_2024_11_12-21_40_57/_FabricModule_e2_i0.pth i=7 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet/ResNet50/no_noise/150+_epochs/run_e150_2024_11_12-21_40_57/_FabricModule_e1_i0.pth i=8 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e500_i0.pth i=9 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e200_i0.pth i=10 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e100_i0.pth i=11 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e50_i0.pth i=12 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e20_i0.pth i=13 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e10_i0.pth i=14 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e5_i0.pth i=15 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e2_i0.pth i=16 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e1_i0.pth i=17 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e400_i0.pth i=18 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e200_i0.pth i=19 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e100_i0.pth i=20 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e50_i0.pth i=21 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e20_i0.pth i=22 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e10_i0.pth i=23 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e5_i0.pth i=24 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e2_i0.pth i=25 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e1_i0.pth i=26 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e200_i0.pth i=27 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e100_i0.pth i=28 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e50_i0.pth i=29 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e20_i0.pth i=30 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e10_i0.pth i=31 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e5_i0.pth i=32 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e2_i0.pth i=33 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e1_i0.pth i=34 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e150_i0.pth i=35 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e100_i0.pth i=36 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e50_i0.pth i=37 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e20_i0.pth i=38 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e10_i0.pth i=39 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e5_i0.pth i=40 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e2_i0.pth i=41 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e1_i0.pth i=42 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e150_i0.pth i=43 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e100_i0.pth i=44 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e50_i0.pth i=45 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e20_i0.pth i=46 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e10_i0.pth i=47 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e5_i0.pth i=48 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e2_i0.pth i=49 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e1_i0.pth i=50 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e1000_i0.pth i=51 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e500_i0.pth i=52 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e200_i0.pth i=53 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e100_i0.pth i=54 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e50_i0.pth i=55 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e20_i0.pth i=56 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e10_i0.pth i=57 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e5_i0.pth i=58 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e2_i0.pth i=59 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e1_i0.pth i=60 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e1000_i0.pth i=61 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e500_i0.pth i=62 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e200_i0.pth i=63 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e100_i0.pth i=64 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e50_i0.pth i=65 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e20_i0.pth i=66 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e10_i0.pth i=67 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e5_i0.pth i=68 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e2_i0.pth i=69 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e1_i0.pth i=70 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_55/ResNet18_32x32_e300_i0.pth i=71 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_55/ResNet18_32x32_e200_i0.pth i=72 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_55/ResNet18_32x32_e100_i0.pth i=73 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_55/ResNet18_32x32_e50_i0.pth i=74 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_55/ResNet18_32x32_e20_i0.pth i=75 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_55/ResNet18_32x32_e10_i0.pth i=76 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_55/ResNet18_32x32_e5_i0.pth i=77 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_55/ResNet18_32x32_e2_i0.pth i=78 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_55/ResNet18_32x32_e1_i0.pth i=79 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_15/ResNet18_32x32_e300_i0.pth i=80 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_15/ResNet18_32x32_e200_i0.pth i=81 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_15/ResNet18_32x32_e100_i0.pth i=82 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_15/ResNet18_32x32_e50_i0.pth i=83 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_15/ResNet18_32x32_e20_i0.pth i=84 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_15/ResNet18_32x32_e10_i0.pth i=85 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_15/ResNet18_32x32_e5_i0.pth i=86 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_15/ResNet18_32x32_e2_i0.pth i=87 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_15/ResNet18_32x32_e1_i0.pth i=88 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_01/ResNet18_32x32_e300_i0.pth i=89 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_01/ResNet18_32x32_e200_i0.pth i=90 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_01/ResNet18_32x32_e100_i0.pth i=91 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_01/ResNet18_32x32_e50_i0.pth i=92 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_01/ResNet18_32x32_e20_i0.pth i=93 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_01/ResNet18_32x32_e10_i0.pth i=94 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_01/ResNet18_32x32_e5_i0.pth i=95 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_01/ResNet18_32x32_e2_i0.pth i=96 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_01/ResNet18_32x32_e1_i0.pth i=97 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_31_03/ResNet18_32x32_e300_i0.pth i=98 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_31_03/ResNet18_32x32_e200_i0.pth i=99 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_31_03/ResNet18_32x32_e100_i0.pth i=100 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_31_03/ResNet18_32x32_e50_i0.pth i=101 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_31_03/ResNet18_32x32_e20_i0.pth i=102 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_31_03/ResNet18_32x32_e10_i0.pth i=103 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_31_03/ResNet18_32x32_e5_i0.pth i=104 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_31_03/ResNet18_32x32_e2_i0.pth i=105 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_31_03/ResNet18_32x32_e1_i0.pth i=106 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_50/ResNet18_32x32_e300_i0.pth i=107 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_50/ResNet18_32x32_e200_i0.pth i=108 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_50/ResNet18_32x32_e100_i0.pth i=109 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_50/ResNet18_32x32_e50_i0.pth i=110 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_50/ResNet18_32x32_e20_i0.pth i=111 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_50/ResNet18_32x32_e10_i0.pth i=112 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_50/ResNet18_32x32_e5_i0.pth i=113 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_50/ResNet18_32x32_e2_i0.pth i=114 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_50/ResNet18_32x32_e1_i0.pth i=115 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_22/ResNet18_32x32_e300_i0.pth i=116 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_22/ResNet18_32x32_e200_i0.pth i=117 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_22/ResNet18_32x32_e100_i0.pth i=118 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_22/ResNet18_32x32_e50_i0.pth i=119 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_22/ResNet18_32x32_e20_i0.pth i=120 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_22/ResNet18_32x32_e10_i0.pth i=121 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_22/ResNet18_32x32_e5_i0.pth i=122 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_22/ResNet18_32x32_e2_i0.pth i=123 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_22/ResNet18_32x32_e1_i0.pth i=124 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_15/ResNet18_32x32_e300_i0.pth i=125 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_15/ResNet18_32x32_e200_i0.pth i=126 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_15/ResNet18_32x32_e100_i0.pth i=127 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_15/ResNet18_32x32_e50_i0.pth i=128 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_15/ResNet18_32x32_e20_i0.pth i=129 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_15/ResNet18_32x32_e10_i0.pth i=130 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_15/ResNet18_32x32_e5_i0.pth i=131 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_15/ResNet18_32x32_e2_i0.pth i=132 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_15/ResNet18_32x32_e1_i0.pth i=133 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_29_36/ResNet18_32x32_e300_i0.pth i=134 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_29_36/ResNet18_32x32_e200_i0.pth i=135 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_29_36/ResNet18_32x32_e100_i0.pth i=136 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_29_36/ResNet18_32x32_e50_i0.pth i=137 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_29_36/ResNet18_32x32_e20_i0.pth i=138 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_29_36/ResNet18_32x32_e10_i0.pth i=139 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_29_36/ResNet18_32x32_e5_i0.pth i=140 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_29_36/ResNet18_32x32_e2_i0.pth i=141 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_29_36/ResNet18_32x32_e1_i0.pth i=142 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_29_04/ResNet18_32x32_e300_i0.pth i=143 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_29_04/ResNet18_32x32_e200_i0.pth i=144 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_29_04/ResNet18_32x32_e100_i0.pth i=145 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_29_04/ResNet18_32x32_e50_i0.pth i=146 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_29_04/ResNet18_32x32_e20_i0.pth i=147 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_29_04/ResNet18_32x32_e10_i0.pth i=148 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_29_04/ResNet18_32x32_e5_i0.pth i=149 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_29_04/ResNet18_32x32_e2_i0.pth i=150 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_29_04/ResNet18_32x32_e1_i0.pth i=151 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_28_40/ResNet18_32x32_e300_i0.pth i=152 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_28_40/ResNet18_32x32_e200_i0.pth i=153 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_28_40/ResNet18_32x32_e100_i0.pth i=154 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_28_40/ResNet18_32x32_e50_i0.pth i=155 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_28_40/ResNet18_32x32_e20_i0.pth i=156 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_28_40/ResNet18_32x32_e10_i0.pth i=157 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_28_40/ResNet18_32x32_e5_i0.pth i=158 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_28_40/ResNet18_32x32_e2_i0.pth i=159 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_28_40/ResNet18_32x32_e1_i0.pth i=160 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_26_42/ResNet18_32x32_e300_i0.pth i=161 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_26_42/ResNet18_32x32_e200_i0.pth i=162 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_26_42/ResNet18_32x32_e100_i0.pth i=163 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_26_42/ResNet18_32x32_e50_i0.pth i=164 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_26_42/ResNet18_32x32_e20_i0.pth i=165 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_26_42/ResNet18_32x32_e10_i0.pth i=166 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_26_42/ResNet18_32x32_e5_i0.pth i=167 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_26_42/ResNet18_32x32_e2_i0.pth i=168 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_26_42/ResNet18_32x32_e1_i0.pth i=169 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_39_58/ResNet18_32x32_e300_i0.pth i=170 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_39_58/ResNet18_32x32_e200_i0.pth i=171 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_39_58/ResNet18_32x32_e100_i0.pth i=172 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_39_58/ResNet18_32x32_e50_i0.pth i=173 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_39_58/ResNet18_32x32_e20_i0.pth i=174 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_39_58/ResNet18_32x32_e10_i0.pth i=175 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_39_58/ResNet18_32x32_e5_i0.pth i=176 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_39_58/ResNet18_32x32_e2_i0.pth i=177 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_39_58/ResNet18_32x32_e1_i0.pth i=178 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_48/ResNet18_32x32_e300_i0.pth i=179 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_48/ResNet18_32x32_e200_i0.pth i=180 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_48/ResNet18_32x32_e100_i0.pth i=181 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_48/ResNet18_32x32_e50_i0.pth i=182 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_48/ResNet18_32x32_e20_i0.pth i=183 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_48/ResNet18_32x32_e10_i0.pth i=184 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_48/ResNet18_32x32_e5_i0.pth i=185 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_48/ResNet18_32x32_e2_i0.pth i=186 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_48/ResNet18_32x32_e1_i0.pth i=187 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_43/ResNet18_32x32_e300_i0.pth i=188 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_43/ResNet18_32x32_e200_i0.pth i=189 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_43/ResNet18_32x32_e100_i0.pth i=190 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_43/ResNet18_32x32_e50_i0.pth i=191 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_43/ResNet18_32x32_e20_i0.pth i=192 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_43/ResNet18_32x32_e10_i0.pth i=193 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_43/ResNet18_32x32_e5_i0.pth i=194 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_43/ResNet18_32x32_e2_i0.pth i=195 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_43/ResNet18_32x32_e1_i0.pth i=196 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_23/ResNet18_32x32_e300_i0.pth i=197 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_23/ResNet18_32x32_e200_i0.pth i=198 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_23/ResNet18_32x32_e100_i0.pth i=199 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_23/ResNet18_32x32_e50_i0.pth i=200 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_23/ResNet18_32x32_e20_i0.pth i=201 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_23/ResNet18_32x32_e10_i0.pth i=202 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_23/ResNet18_32x32_e5_i0.pth i=203 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_23/ResNet18_32x32_e2_i0.pth i=204 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_23/ResNet18_32x32_e1_i0.pth i=205 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_56/ResNet18_32x32_e300_i0.pth i=206 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_56/ResNet18_32x32_e200_i0.pth i=207 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_56/ResNet18_32x32_e100_i0.pth i=208 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_56/ResNet18_32x32_e50_i0.pth i=209 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_56/ResNet18_32x32_e20_i0.pth i=210 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_56/ResNet18_32x32_e10_i0.pth i=211 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_56/ResNet18_32x32_e5_i0.pth i=212 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_56/ResNet18_32x32_e2_i0.pth i=213 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_56/ResNet18_32x32_e1_i0.pth i=214 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_37/ResNet18_32x32_e300_i0.pth i=215 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_37/ResNet18_32x32_e200_i0.pth i=216 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_37/ResNet18_32x32_e100_i0.pth i=217 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_37/ResNet18_32x32_e50_i0.pth i=218 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_37/ResNet18_32x32_e20_i0.pth i=219 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_37/ResNet18_32x32_e10_i0.pth i=220 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_37/ResNet18_32x32_e5_i0.pth i=221 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_37/ResNet18_32x32_e2_i0.pth i=222 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_37/ResNet18_32x32_e1_i0.pth i=223 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_29/ResNet18_32x32_e300_i0.pth i=224 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_29/ResNet18_32x32_e200_i0.pth i=225 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_29/ResNet18_32x32_e100_i0.pth i=226 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_29/ResNet18_32x32_e50_i0.pth i=227 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_29/ResNet18_32x32_e20_i0.pth i=228 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_29/ResNet18_32x32_e10_i0.pth i=229 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_29/ResNet18_32x32_e5_i0.pth i=230 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_29/ResNet18_32x32_e2_i0.pth i=231 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_29/ResNet18_32x32_e1_i0.pth i=232 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_15/ResNet18_32x32_e300_i0.pth i=233 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_15/ResNet18_32x32_e200_i0.pth i=234 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_15/ResNet18_32x32_e100_i0.pth i=235 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_15/ResNet18_32x32_e50_i0.pth i=236 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_15/ResNet18_32x32_e20_i0.pth i=237 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_15/ResNet18_32x32_e10_i0.pth i=238 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_15/ResNet18_32x32_e5_i0.pth i=239 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_15/ResNet18_32x32_e2_i0.pth i=240 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_15/ResNet18_32x32_e1_i0.pth i=241 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_54/ResNet18_32x32_e300_i0.pth i=242 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_54/ResNet18_32x32_e200_i0.pth i=243 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_54/ResNet18_32x32_e100_i0.pth i=244 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_54/ResNet18_32x32_e50_i0.pth i=245 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_54/ResNet18_32x32_e20_i0.pth i=246 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_54/ResNet18_32x32_e10_i0.pth i=247 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_54/ResNet18_32x32_e5_i0.pth i=248 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_54/ResNet18_32x32_e2_i0.pth i=249 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_54/ResNet18_32x32_e1_i0.pth i=250 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_46/ResNet18_32x32_e300_i0.pth i=251 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_46/ResNet18_32x32_e200_i0.pth i=252 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_46/ResNet18_32x32_e100_i0.pth i=253 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_46/ResNet18_32x32_e50_i0.pth i=254 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_46/ResNet18_32x32_e20_i0.pth i=255 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_46/ResNet18_32x32_e10_i0.pth i=256 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_46/ResNet18_32x32_e5_i0.pth i=257 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_46/ResNet18_32x32_e2_i0.pth i=258 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_46/ResNet18_32x32_e1_i0.pth i=259 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_44/ResNet18_32x32_e300_i0.pth i=260 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_44/ResNet18_32x32_e200_i0.pth i=261 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_44/ResNet18_32x32_e100_i0.pth i=262 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_44/ResNet18_32x32_e50_i0.pth i=263 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_44/ResNet18_32x32_e20_i0.pth i=264 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_44/ResNet18_32x32_e10_i0.pth i=265 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_44/ResNet18_32x32_e5_i0.pth i=266 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_44/ResNet18_32x32_e2_i0.pth i=267 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_44/ResNet18_32x32_e1_i0.pth i=268 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_30/ResNet18_32x32_e300_i0.pth i=269 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_30/ResNet18_32x32_e200_i0.pth i=270 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_30/ResNet18_32x32_e100_i0.pth i=271 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_30/ResNet18_32x32_e50_i0.pth i=272 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_30/ResNet18_32x32_e20_i0.pth i=273 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_30/ResNet18_32x32_e10_i0.pth i=274 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_30/ResNet18_32x32_e5_i0.pth i=275 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_30/ResNet18_32x32_e2_i0.pth i=276 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_30/ResNet18_32x32_e1_i0.pth i=277 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_33_15/ResNet18_32x32_e300_i0.pth i=278 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_33_15/ResNet18_32x32_e200_i0.pth i=279 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_33_15/ResNet18_32x32_e100_i0.pth i=280 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_33_15/ResNet18_32x32_e50_i0.pth i=281 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_33_15/ResNet18_32x32_e20_i0.pth i=282 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_33_15/ResNet18_32x32_e10_i0.pth i=283 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_33_15/ResNet18_32x32_e5_i0.pth i=284 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_33_15/ResNet18_32x32_e2_i0.pth i=285 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_33_15/ResNet18_32x32_e1_i0.pth i=286 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_46_33/ResNet18_32x32_e300_i0.pth i=287 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_46_33/ResNet18_32x32_e200_i0.pth i=288 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_46_33/ResNet18_32x32_e100_i0.pth i=289 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_46_33/ResNet18_32x32_e50_i0.pth i=290 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_46_33/ResNet18_32x32_e20_i0.pth i=291 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_46_33/ResNet18_32x32_e10_i0.pth i=292 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_46_33/ResNet18_32x32_e5_i0.pth i=293 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_46_33/ResNet18_32x32_e2_i0.pth i=294 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_46_33/ResNet18_32x32_e1_i0.pth i=295 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_35/ResNet18_32x32_e300_i0.pth i=296 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_35/ResNet18_32x32_e200_i0.pth i=297 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_35/ResNet18_32x32_e100_i0.pth i=298 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_35/ResNet18_32x32_e50_i0.pth i=299 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_35/ResNet18_32x32_e20_i0.pth i=300 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_35/ResNet18_32x32_e10_i0.pth i=301 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_35/ResNet18_32x32_e5_i0.pth i=302 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_35/ResNet18_32x32_e2_i0.pth i=303 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_35/ResNet18_32x32_e1_i0.pth i=304 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_19/ResNet18_32x32_e300_i0.pth i=305 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_19/ResNet18_32x32_e200_i0.pth i=306 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_19/ResNet18_32x32_e100_i0.pth i=307 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_19/ResNet18_32x32_e50_i0.pth i=308 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_19/ResNet18_32x32_e20_i0.pth i=309 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_19/ResNet18_32x32_e10_i0.pth i=310 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_19/ResNet18_32x32_e5_i0.pth i=311 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_19/ResNet18_32x32_e2_i0.pth i=312 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_19/ResNet18_32x32_e1_i0.pth i=313 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_09/ResNet18_32x32_e300_i0.pth i=314 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_09/ResNet18_32x32_e200_i0.pth i=315 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_09/ResNet18_32x32_e100_i0.pth i=316 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_09/ResNet18_32x32_e50_i0.pth i=317 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_09/ResNet18_32x32_e20_i0.pth i=318 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_09/ResNet18_32x32_e10_i0.pth i=319 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_09/ResNet18_32x32_e5_i0.pth i=320 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_09/ResNet18_32x32_e2_i0.pth i=321 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_09/ResNet18_32x32_e1_i0.pth i=322 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_44_09/ResNet18_32x32_e300_i0.pth i=323 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_44_09/ResNet18_32x32_e200_i0.pth i=324 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_44_09/ResNet18_32x32_e100_i0.pth i=325 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_44_09/ResNet18_32x32_e50_i0.pth i=326 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_44_09/ResNet18_32x32_e20_i0.pth i=327 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_44_09/ResNet18_32x32_e10_i0.pth i=328 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_44_09/ResNet18_32x32_e5_i0.pth i=329 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_44_09/ResNet18_32x32_e2_i0.pth i=330 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_44_09/ResNet18_32x32_e1_i0.pth i=331 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_34/ResNet18_32x32_e300_i0.pth i=332 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_34/ResNet18_32x32_e200_i0.pth i=333 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_34/ResNet18_32x32_e100_i0.pth i=334 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_34/ResNet18_32x32_e50_i0.pth i=335 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_34/ResNet18_32x32_e20_i0.pth i=336 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_34/ResNet18_32x32_e10_i0.pth i=337 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_34/ResNet18_32x32_e5_i0.pth i=338 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_34/ResNet18_32x32_e2_i0.pth i=339 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_34/ResNet18_32x32_e1_i0.pth i=340 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_12/ResNet18_32x32_e300_i0.pth i=341 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_12/ResNet18_32x32_e200_i0.pth i=342 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_12/ResNet18_32x32_e100_i0.pth i=343 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_12/ResNet18_32x32_e50_i0.pth i=344 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_12/ResNet18_32x32_e20_i0.pth i=345 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_12/ResNet18_32x32_e10_i0.pth i=346 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_12/ResNet18_32x32_e5_i0.pth i=347 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_12/ResNet18_32x32_e2_i0.pth i=348 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_12/ResNet18_32x32_e1_i0.pth i=349 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_09/ResNet18_32x32_e300_i0.pth i=350 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_09/ResNet18_32x32_e200_i0.pth i=351 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_09/ResNet18_32x32_e100_i0.pth i=352 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_09/ResNet18_32x32_e50_i0.pth i=353 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_09/ResNet18_32x32_e20_i0.pth i=354 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_09/ResNet18_32x32_e10_i0.pth i=355 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_09/ResNet18_32x32_e5_i0.pth i=356 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_09/ResNet18_32x32_e2_i0.pth i=357 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_09/ResNet18_32x32_e1_i0.pth i=358 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_42_38/ResNet18_32x32_e300_i0.pth i=359 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_42_38/ResNet18_32x32_e200_i0.pth i=360 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_42_38/ResNet18_32x32_e100_i0.pth i=361 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_42_38/ResNet18_32x32_e50_i0.pth i=362 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_42_38/ResNet18_32x32_e20_i0.pth i=363 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_42_38/ResNet18_32x32_e10_i0.pth i=364 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_42_38/ResNet18_32x32_e5_i0.pth i=365 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_42_38/ResNet18_32x32_e2_i0.pth i=366 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_42_38/ResNet18_32x32_e1_i0.pth i=367 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_42_37/ResNet18_32x32_e300_i0.pth i=368 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_42_37/ResNet18_32x32_e200_i0.pth i=369 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_42_37/ResNet18_32x32_e100_i0.pth i=370 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_42_37/ResNet18_32x32_e50_i0.pth i=371 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_42_37/ResNet18_32x32_e20_i0.pth i=372 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_42_37/ResNet18_32x32_e10_i0.pth i=373 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_42_37/ResNet18_32x32_e5_i0.pth i=374 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_42_37/ResNet18_32x32_e2_i0.pth i=375 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_42_37/ResNet18_32x32_e1_i0.pth i=376 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_41/ResNet18_32x32_e300_i0.pth i=377 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_41/ResNet18_32x32_e200_i0.pth i=378 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_41/ResNet18_32x32_e100_i0.pth i=379 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_41/ResNet18_32x32_e50_i0.pth i=380 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_41/ResNet18_32x32_e20_i0.pth i=381 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_41/ResNet18_32x32_e10_i0.pth i=382 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_41/ResNet18_32x32_e5_i0.pth i=383 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_41/ResNet18_32x32_e2_i0.pth i=384 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_41/ResNet18_32x32_e1_i0.pth i=385 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_30/ResNet18_32x32_e300_i0.pth i=386 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_30/ResNet18_32x32_e200_i0.pth i=387 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_30/ResNet18_32x32_e100_i0.pth i=388 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_30/ResNet18_32x32_e50_i0.pth i=389 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_30/ResNet18_32x32_e20_i0.pth i=390 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_30/ResNet18_32x32_e10_i0.pth i=391 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_30/ResNet18_32x32_e5_i0.pth i=392 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_30/ResNet18_32x32_e2_i0.pth i=393 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_30/ResNet18_32x32_e1_i0.pth i=394 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_22/ResNet18_32x32_e300_i0.pth i=395 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_22/ResNet18_32x32_e200_i0.pth i=396 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_22/ResNet18_32x32_e100_i0.pth i=397 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_22/ResNet18_32x32_e50_i0.pth i=398 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_22/ResNet18_32x32_e20_i0.pth i=399 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_22/ResNet18_32x32_e10_i0.pth i=400 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_22/ResNet18_32x32_e5_i0.pth i=401 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_22/ResNet18_32x32_e2_i0.pth i=402 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_22/ResNet18_32x32_e1_i0.pth i=403 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_40_49/ResNet18_32x32_e300_i0.pth i=404 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_40_49/ResNet18_32x32_e200_i0.pth i=405 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_40_49/ResNet18_32x32_e100_i0.pth i=406 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_40_49/ResNet18_32x32_e50_i0.pth i=407 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_40_49/ResNet18_32x32_e20_i0.pth i=408 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_40_49/ResNet18_32x32_e10_i0.pth i=409 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_40_49/ResNet18_32x32_e5_i0.pth i=410 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_40_49/ResNet18_32x32_e2_i0.pth i=411 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_40_49/ResNet18_32x32_e1_i0.pth i=412 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_40_10/ResNet18_32x32_e300_i0.pth i=413 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_40_10/ResNet18_32x32_e200_i0.pth i=414 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_40_10/ResNet18_32x32_e100_i0.pth i=415 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_40_10/ResNet18_32x32_e50_i0.pth i=416 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_40_10/ResNet18_32x32_e20_i0.pth i=417 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_40_10/ResNet18_32x32_e10_i0.pth i=418 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_40_10/ResNet18_32x32_e5_i0.pth i=419 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_40_10/ResNet18_32x32_e2_i0.pth i=420 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_40_10/ResNet18_32x32_e1_i0.pth i=421 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_34/ResNet18_32x32_e300_i0.pth i=422 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_34/ResNet18_32x32_e200_i0.pth i=423 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_34/ResNet18_32x32_e100_i0.pth i=424 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_34/ResNet18_32x32_e50_i0.pth i=425 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_34/ResNet18_32x32_e20_i0.pth i=426 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_34/ResNet18_32x32_e10_i0.pth i=427 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_34/ResNet18_32x32_e5_i0.pth i=428 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_34/ResNet18_32x32_e2_i0.pth i=429 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_34/ResNet18_32x32_e1_i0.pth i=430 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_23/ResNet18_32x32_e300_i0.pth i=431 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_23/ResNet18_32x32_e200_i0.pth i=432 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_23/ResNet18_32x32_e100_i0.pth i=433 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_23/ResNet18_32x32_e50_i0.pth i=434 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_23/ResNet18_32x32_e20_i0.pth i=435 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_23/ResNet18_32x32_e10_i0.pth i=436 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_23/ResNet18_32x32_e5_i0.pth i=437 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_23/ResNet18_32x32_e2_i0.pth i=438 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_23/ResNet18_32x32_e1_i0.pth i=439 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_15/ResNet18_32x32_e300_i0.pth i=440 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_15/ResNet18_32x32_e200_i0.pth i=441 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_15/ResNet18_32x32_e100_i0.pth i=442 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_15/ResNet18_32x32_e50_i0.pth i=443 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_15/ResNet18_32x32_e20_i0.pth i=444 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_15/ResNet18_32x32_e10_i0.pth i=445 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_15/ResNet18_32x32_e5_i0.pth i=446 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_15/ResNet18_32x32_e2_i0.pth i=447 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_15/ResNet18_32x32_e1_i0.pth i=448 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_50_50/ResNet18_32x32_e300_i0.pth i=449 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_50_50/ResNet18_32x32_e200_i0.pth i=450 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_50_50/ResNet18_32x32_e100_i0.pth i=451 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_50_50/ResNet18_32x32_e50_i0.pth i=452 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_50_50/ResNet18_32x32_e20_i0.pth i=453 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_50_50/ResNet18_32x32_e10_i0.pth i=454 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_50_50/ResNet18_32x32_e5_i0.pth i=455 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_50_50/ResNet18_32x32_e2_i0.pth i=456 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_50_50/ResNet18_32x32_e1_i0.pth i=457 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_50_04/ResNet18_32x32_e300_i0.pth i=458 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_50_04/ResNet18_32x32_e200_i0.pth i=459 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_50_04/ResNet18_32x32_e100_i0.pth i=460 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_50_04/ResNet18_32x32_e50_i0.pth i=461 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_50_04/ResNet18_32x32_e20_i0.pth i=462 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_50_04/ResNet18_32x32_e10_i0.pth i=463 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_50_04/ResNet18_32x32_e5_i0.pth i=464 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_50_04/ResNet18_32x32_e2_i0.pth i=465 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_50_04/ResNet18_32x32_e1_i0.pth i=466 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_50/ResNet18_32x32_e300_i0.pth i=467 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_50/ResNet18_32x32_e200_i0.pth i=468 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_50/ResNet18_32x32_e100_i0.pth i=469 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_50/ResNet18_32x32_e50_i0.pth i=470 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_50/ResNet18_32x32_e20_i0.pth i=471 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_50/ResNet18_32x32_e10_i0.pth i=472 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_50/ResNet18_32x32_e5_i0.pth i=473 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_50/ResNet18_32x32_e2_i0.pth i=474 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_50/ResNet18_32x32_e1_i0.pth i=475 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_49/ResNet18_32x32_e300_i0.pth i=476 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_49/ResNet18_32x32_e200_i0.pth i=477 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_49/ResNet18_32x32_e100_i0.pth i=478 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_49/ResNet18_32x32_e50_i0.pth i=479 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_49/ResNet18_32x32_e20_i0.pth i=480 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_49/ResNet18_32x32_e10_i0.pth i=481 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_49/ResNet18_32x32_e5_i0.pth i=482 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_49/ResNet18_32x32_e2_i0.pth i=483 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_49/ResNet18_32x32_e1_i0.pth i=484 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_36/ResNet18_32x32_e300_i0.pth i=485 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_36/ResNet18_32x32_e200_i0.pth i=486 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_36/ResNet18_32x32_e100_i0.pth i=487 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_36/ResNet18_32x32_e50_i0.pth i=488 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_36/ResNet18_32x32_e20_i0.pth i=489 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_36/ResNet18_32x32_e10_i0.pth i=490 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_36/ResNet18_32x32_e5_i0.pth i=491 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_36/ResNet18_32x32_e2_i0.pth i=492 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_36/ResNet18_32x32_e1_i0.pth i=493 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_20/ResNet18_32x32_e300_i0.pth i=494 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_20/ResNet18_32x32_e200_i0.pth i=495 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_20/ResNet18_32x32_e100_i0.pth i=496 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_20/ResNet18_32x32_e50_i0.pth i=497 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_20/ResNet18_32x32_e20_i0.pth i=498 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_20/ResNet18_32x32_e10_i0.pth i=499 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_20/ResNet18_32x32_e5_i0.pth i=500 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_20/ResNet18_32x32_e2_i0.pth i=501 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_20/ResNet18_32x32_e1_i0.pth i=502 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_44/ResNet18_32x32_e300_i0.pth i=503 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_44/ResNet18_32x32_e200_i0.pth i=504 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_44/ResNet18_32x32_e100_i0.pth i=505 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_44/ResNet18_32x32_e50_i0.pth i=506 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_44/ResNet18_32x32_e20_i0.pth i=507 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_44/ResNet18_32x32_e10_i0.pth i=508 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_44/ResNet18_32x32_e5_i0.pth i=509 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_44/ResNet18_32x32_e2_i0.pth i=510 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_44/ResNet18_32x32_e1_i0.pth i=511 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_13/ResNet18_32x32_e300_i0.pth i=512 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_13/ResNet18_32x32_e200_i0.pth i=513 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_13/ResNet18_32x32_e100_i0.pth i=514 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_13/ResNet18_32x32_e50_i0.pth i=515 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_13/ResNet18_32x32_e20_i0.pth i=516 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_13/ResNet18_32x32_e10_i0.pth i=517 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_13/ResNet18_32x32_e5_i0.pth i=518 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_13/ResNet18_32x32_e2_i0.pth i=519 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_13/ResNet18_32x32_e1_i0.pth i=520 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_11/ResNet18_32x32_e300_i0.pth i=521 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_11/ResNet18_32x32_e200_i0.pth i=522 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_11/ResNet18_32x32_e100_i0.pth i=523 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_11/ResNet18_32x32_e50_i0.pth i=524 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_11/ResNet18_32x32_e20_i0.pth i=525 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_11/ResNet18_32x32_e10_i0.pth i=526 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_11/ResNet18_32x32_e5_i0.pth i=527 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_11/ResNet18_32x32_e2_i0.pth i=528 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_11/ResNet18_32x32_e1_i0.pth i=529 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_47_48/ResNet18_32x32_e300_i0.pth i=530 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_47_48/ResNet18_32x32_e200_i0.pth i=531 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_47_48/ResNet18_32x32_e100_i0.pth i=532 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_47_48/ResNet18_32x32_e50_i0.pth i=533 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_47_48/ResNet18_32x32_e20_i0.pth i=534 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_47_48/ResNet18_32x32_e10_i0.pth i=535 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_47_48/ResNet18_32x32_e5_i0.pth i=536 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_47_48/ResNet18_32x32_e2_i0.pth i=537 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_47_48/ResNet18_32x32_e1_i0.pth i=538 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_47_08/ResNet18_32x32_e300_i0.pth i=539 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_47_08/ResNet18_32x32_e200_i0.pth i=540 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_47_08/ResNet18_32x32_e100_i0.pth i=541 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_47_08/ResNet18_32x32_e50_i0.pth i=542 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_47_08/ResNet18_32x32_e20_i0.pth i=543 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_47_08/ResNet18_32x32_e10_i0.pth i=544 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_47_08/ResNet18_32x32_e5_i0.pth i=545 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_47_08/ResNet18_32x32_e2_i0.pth i=546 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_47_08/ResNet18_32x32_e1_i0.pth i=547 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_46_37/ResNet18_32x32_e300_i0.pth i=548 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_46_37/ResNet18_32x32_e200_i0.pth i=549 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_46_37/ResNet18_32x32_e100_i0.pth i=550 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_46_37/ResNet18_32x32_e50_i0.pth i=551 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_46_37/ResNet18_32x32_e20_i0.pth i=552 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_46_37/ResNet18_32x32_e10_i0.pth i=553 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_46_37/ResNet18_32x32_e5_i0.pth i=554 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_46_37/ResNet18_32x32_e2_i0.pth i=555 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_46_37/ResNet18_32x32_e1_i0.pth i=556 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_59_49/ResNet18_32x32_e300_i0.pth i=557 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_59_49/ResNet18_32x32_e200_i0.pth i=558 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_59_49/ResNet18_32x32_e100_i0.pth i=559 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_59_49/ResNet18_32x32_e50_i0.pth i=560 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_59_49/ResNet18_32x32_e20_i0.pth i=561 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_59_49/ResNet18_32x32_e10_i0.pth i=562 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_59_49/ResNet18_32x32_e5_i0.pth i=563 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_59_49/ResNet18_32x32_e2_i0.pth i=564 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_59_49/ResNet18_32x32_e1_i0.pth i=565 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_59_22/ResNet18_32x32_e300_i0.pth i=566 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_59_22/ResNet18_32x32_e200_i0.pth i=567 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_59_22/ResNet18_32x32_e100_i0.pth i=568 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_59_22/ResNet18_32x32_e50_i0.pth i=569 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_59_22/ResNet18_32x32_e20_i0.pth i=570 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_59_22/ResNet18_32x32_e10_i0.pth i=571 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_59_22/ResNet18_32x32_e5_i0.pth i=572 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_59_22/ResNet18_32x32_e2_i0.pth i=573 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_59_22/ResNet18_32x32_e1_i0.pth i=574 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_58_34/ResNet18_32x32_e300_i0.pth i=575 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_58_34/ResNet18_32x32_e200_i0.pth i=576 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_58_34/ResNet18_32x32_e100_i0.pth i=577 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_58_34/ResNet18_32x32_e50_i0.pth i=578 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_58_34/ResNet18_32x32_e20_i0.pth i=579 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_58_34/ResNet18_32x32_e10_i0.pth i=580 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_58_34/ResNet18_32x32_e5_i0.pth i=581 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_58_34/ResNet18_32x32_e2_i0.pth i=582 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_58_34/ResNet18_32x32_e1_i0.pth i=583 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_58_05/ResNet18_32x32_e300_i0.pth i=584 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_58_05/ResNet18_32x32_e200_i0.pth i=585 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_58_05/ResNet18_32x32_e100_i0.pth i=586 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_58_05/ResNet18_32x32_e50_i0.pth i=587 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_58_05/ResNet18_32x32_e20_i0.pth i=588 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_58_05/ResNet18_32x32_e10_i0.pth i=589 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_58_05/ResNet18_32x32_e5_i0.pth i=590 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_58_05/ResNet18_32x32_e2_i0.pth i=591 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_58_05/ResNet18_32x32_e1_i0.pth i=592 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_57_33/ResNet18_32x32_e300_i0.pth i=593 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_57_33/ResNet18_32x32_e200_i0.pth i=594 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_57_33/ResNet18_32x32_e100_i0.pth i=595 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_57_33/ResNet18_32x32_e50_i0.pth i=596 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_57_33/ResNet18_32x32_e20_i0.pth i=597 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_57_33/ResNet18_32x32_e10_i0.pth i=598 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_57_33/ResNet18_32x32_e5_i0.pth i=599 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_57_33/ResNet18_32x32_e2_i0.pth i=600 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_57_33/ResNet18_32x32_e1_i0.pth i=601 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_57_22/ResNet18_32x32_e300_i0.pth i=602 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_57_22/ResNet18_32x32_e200_i0.pth i=603 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_57_22/ResNet18_32x32_e100_i0.pth i=604 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_57_22/ResNet18_32x32_e50_i0.pth i=605 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_57_22/ResNet18_32x32_e20_i0.pth i=606 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_57_22/ResNet18_32x32_e10_i0.pth i=607 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_57_22/ResNet18_32x32_e5_i0.pth i=608 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_57_22/ResNet18_32x32_e2_i0.pth i=609 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_57_22/ResNet18_32x32_e1_i0.pth i=610 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_56_54/ResNet18_32x32_e300_i0.pth i=611 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_56_54/ResNet18_32x32_e200_i0.pth i=612 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_56_54/ResNet18_32x32_e100_i0.pth i=613 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_56_54/ResNet18_32x32_e50_i0.pth i=614 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_56_54/ResNet18_32x32_e20_i0.pth i=615 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_56_54/ResNet18_32x32_e10_i0.pth i=616 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_56_54/ResNet18_32x32_e5_i0.pth i=617 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_56_54/ResNet18_32x32_e2_i0.pth i=618 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_56_54/ResNet18_32x32_e1_i0.pth i=619 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_56_19/ResNet18_32x32_e300_i0.pth i=620 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_56_19/ResNet18_32x32_e200_i0.pth i=621 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_56_19/ResNet18_32x32_e100_i0.pth i=622 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_56_19/ResNet18_32x32_e50_i0.pth i=623 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_56_19/ResNet18_32x32_e20_i0.pth i=624 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_56_19/ResNet18_32x32_e10_i0.pth i=625 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_56_19/ResNet18_32x32_e5_i0.pth i=626 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_56_19/ResNet18_32x32_e2_i0.pth i=627 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_56_19/ResNet18_32x32_e1_i0.pth i=628 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_47/ResNet18_32x32_e300_i0.pth i=629 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_47/ResNet18_32x32_e200_i0.pth i=630 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_47/ResNet18_32x32_e100_i0.pth i=631 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_47/ResNet18_32x32_e50_i0.pth i=632 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_47/ResNet18_32x32_e20_i0.pth i=633 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_47/ResNet18_32x32_e10_i0.pth i=634 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_47/ResNet18_32x32_e5_i0.pth i=635 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_47/ResNet18_32x32_e2_i0.pth i=636 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_47/ResNet18_32x32_e1_i0.pth i=637 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_30/ResNet18_32x32_e300_i0.pth i=638 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_30/ResNet18_32x32_e200_i0.pth i=639 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_30/ResNet18_32x32_e100_i0.pth i=640 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_30/ResNet18_32x32_e50_i0.pth i=641 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_30/ResNet18_32x32_e20_i0.pth i=642 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_30/ResNet18_32x32_e10_i0.pth i=643 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_30/ResNet18_32x32_e5_i0.pth i=644 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_30/ResNet18_32x32_e2_i0.pth i=645 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_30/ResNet18_32x32_e1_i0.pth i=646 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_28/ResNet18_32x32_e300_i0.pth i=647 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_28/ResNet18_32x32_e200_i0.pth i=648 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_28/ResNet18_32x32_e100_i0.pth i=649 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_28/ResNet18_32x32_e50_i0.pth i=650 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_28/ResNet18_32x32_e20_i0.pth i=651 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_28/ResNet18_32x32_e10_i0.pth i=652 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_28/ResNet18_32x32_e5_i0.pth i=653 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_28/ResNet18_32x32_e2_i0.pth i=654 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_28/ResNet18_32x32_e1_i0.pth i=655 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_20/ResNet18_32x32_e300_i0.pth i=656 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_20/ResNet18_32x32_e200_i0.pth i=657 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_20/ResNet18_32x32_e100_i0.pth i=658 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_20/ResNet18_32x32_e50_i0.pth i=659 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_20/ResNet18_32x32_e20_i0.pth i=660 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_20/ResNet18_32x32_e10_i0.pth i=661 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_20/ResNet18_32x32_e5_i0.pth i=662 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_20/ResNet18_32x32_e2_i0.pth i=663 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_20/ResNet18_32x32_e1_i0.pth i=664 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_15/ResNet18_32x32_e300_i0.pth i=665 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_15/ResNet18_32x32_e200_i0.pth i=666 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_15/ResNet18_32x32_e100_i0.pth i=667 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_15/ResNet18_32x32_e50_i0.pth i=668 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_15/ResNet18_32x32_e20_i0.pth i=669 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_15/ResNet18_32x32_e10_i0.pth i=670 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_15/ResNet18_32x32_e5_i0.pth i=671 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_15/ResNet18_32x32_e2_i0.pth i=672 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_15/ResNet18_32x32_e1_i0.pth i=673 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_54_33/ResNet18_32x32_e300_i0.pth i=674 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_54_33/ResNet18_32x32_e200_i0.pth i=675 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_54_33/ResNet18_32x32_e100_i0.pth i=676 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_54_33/ResNet18_32x32_e50_i0.pth i=677 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_54_33/ResNet18_32x32_e20_i0.pth i=678 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_54_33/ResNet18_32x32_e10_i0.pth i=679 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_54_33/ResNet18_32x32_e5_i0.pth i=680 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_54_33/ResNet18_32x32_e2_i0.pth i=681 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_54_33/ResNet18_32x32_e1_i0.pth i=682 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_54_26/ResNet18_32x32_e300_i0.pth i=683 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_54_26/ResNet18_32x32_e200_i0.pth i=684 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_54_26/ResNet18_32x32_e100_i0.pth i=685 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_54_26/ResNet18_32x32_e50_i0.pth i=686 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_54_26/ResNet18_32x32_e20_i0.pth i=687 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_54_26/ResNet18_32x32_e10_i0.pth i=688 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_54_26/ResNet18_32x32_e5_i0.pth i=689 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_54_26/ResNet18_32x32_e2_i0.pth i=690 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_54_26/ResNet18_32x32_e1_i0.pth i=691 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_03_13/ResNet18_32x32_e300_i0.pth i=692 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_03_13/ResNet18_32x32_e200_i0.pth i=693 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_03_13/ResNet18_32x32_e100_i0.pth i=694 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_03_13/ResNet18_32x32_e50_i0.pth i=695 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_03_13/ResNet18_32x32_e20_i0.pth i=696 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_03_13/ResNet18_32x32_e10_i0.pth i=697 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_03_13/ResNet18_32x32_e5_i0.pth i=698 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_03_13/ResNet18_32x32_e2_i0.pth i=699 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_03_13/ResNet18_32x32_e1_i0.pth i=700 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_03_12/ResNet18_32x32_e300_i0.pth i=701 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_03_12/ResNet18_32x32_e200_i0.pth i=702 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_03_12/ResNet18_32x32_e100_i0.pth i=703 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_03_12/ResNet18_32x32_e50_i0.pth i=704 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_03_12/ResNet18_32x32_e20_i0.pth i=705 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_03_12/ResNet18_32x32_e10_i0.pth i=706 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_03_12/ResNet18_32x32_e5_i0.pth i=707 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_03_12/ResNet18_32x32_e2_i0.pth i=708 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_03_12/ResNet18_32x32_e1_i0.pth i=709 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_01_32/ResNet18_32x32_e300_i0.pth i=710 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_01_32/ResNet18_32x32_e200_i0.pth i=711 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_01_32/ResNet18_32x32_e100_i0.pth i=712 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_01_32/ResNet18_32x32_e50_i0.pth i=713 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_01_32/ResNet18_32x32_e20_i0.pth i=714 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_01_32/ResNet18_32x32_e10_i0.pth i=715 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_01_32/ResNet18_32x32_e5_i0.pth i=716 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_01_32/ResNet18_32x32_e2_i0.pth i=717 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_01_32/ResNet18_32x32_e1_i0.pth i=718 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_01_00/ResNet18_32x32_e300_i0.pth i=719 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_01_00/ResNet18_32x32_e200_i0.pth i=720 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_01_00/ResNet18_32x32_e100_i0.pth i=721 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_01_00/ResNet18_32x32_e50_i0.pth i=722 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_01_00/ResNet18_32x32_e20_i0.pth i=723 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_01_00/ResNet18_32x32_e10_i0.pth i=724 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_01_00/ResNet18_32x32_e5_i0.pth i=725 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_01_00/ResNet18_32x32_e2_i0.pth i=726 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_01_00/ResNet18_32x32_e1_i0.pth i=727 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_40/ResNet18_32x32_e300_i0.pth i=728 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_40/ResNet18_32x32_e200_i0.pth i=729 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_40/ResNet18_32x32_e100_i0.pth i=730 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_40/ResNet18_32x32_e50_i0.pth i=731 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_40/ResNet18_32x32_e20_i0.pth i=732 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_40/ResNet18_32x32_e10_i0.pth i=733 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_40/ResNet18_32x32_e5_i0.pth i=734 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_40/ResNet18_32x32_e2_i0.pth i=735 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_40/ResNet18_32x32_e1_i0.pth i=736 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_16/ResNet18_32x32_e300_i0.pth i=737 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_16/ResNet18_32x32_e200_i0.pth i=738 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_16/ResNet18_32x32_e100_i0.pth i=739 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_16/ResNet18_32x32_e50_i0.pth i=740 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_16/ResNet18_32x32_e20_i0.pth i=741 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_16/ResNet18_32x32_e10_i0.pth i=742 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_16/ResNet18_32x32_e5_i0.pth i=743 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_16/ResNet18_32x32_e2_i0.pth i=744 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_16/ResNet18_32x32_e1_i0.pth i=745 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_07/ResNet18_32x32_e300_i0.pth i=746 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_07/ResNet18_32x32_e200_i0.pth i=747 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_07/ResNet18_32x32_e100_i0.pth i=748 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_07/ResNet18_32x32_e50_i0.pth i=749 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_07/ResNet18_32x32_e20_i0.pth i=750 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_07/ResNet18_32x32_e10_i0.pth i=751 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_07/ResNet18_32x32_e5_i0.pth i=752 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_07/ResNet18_32x32_e2_i0.pth i=753 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_07/ResNet18_32x32_e1_i0.pth i=754 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_59_13/ResNet18_32x32_e300_i0.pth i=755 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_59_13/ResNet18_32x32_e200_i0.pth i=756 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_59_13/ResNet18_32x32_e100_i0.pth i=757 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_59_13/ResNet18_32x32_e50_i0.pth i=758 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_59_13/ResNet18_32x32_e20_i0.pth i=759 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_59_13/ResNet18_32x32_e10_i0.pth i=760 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_59_13/ResNet18_32x32_e5_i0.pth i=761 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_59_13/ResNet18_32x32_e2_i0.pth i=762 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_59_13/ResNet18_32x32_e1_i0.pth i=763 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_58_48/ResNet18_32x32_e300_i0.pth i=764 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_58_48/ResNet18_32x32_e200_i0.pth i=765 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_58_48/ResNet18_32x32_e100_i0.pth i=766 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_58_48/ResNet18_32x32_e50_i0.pth i=767 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_58_48/ResNet18_32x32_e20_i0.pth i=768 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_58_48/ResNet18_32x32_e10_i0.pth i=769 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_58_48/ResNet18_32x32_e5_i0.pth i=770 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_58_48/ResNet18_32x32_e2_i0.pth i=771 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_58_48/ResNet18_32x32_e1_i0.pth i=772 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_58_40/ResNet18_32x32_e300_i0.pth i=773 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_58_40/ResNet18_32x32_e200_i0.pth i=774 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_58_40/ResNet18_32x32_e100_i0.pth i=775 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_58_40/ResNet18_32x32_e50_i0.pth i=776 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_58_40/ResNet18_32x32_e20_i0.pth i=777 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_58_40/ResNet18_32x32_e10_i0.pth i=778 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_58_40/ResNet18_32x32_e5_i0.pth i=779 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_58_40/ResNet18_32x32_e2_i0.pth i=780 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_58_40/ResNet18_32x32_e1_i0.pth i=781 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_56/ResNet18_32x32_e300_i0.pth i=782 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_56/ResNet18_32x32_e200_i0.pth i=783 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_56/ResNet18_32x32_e100_i0.pth i=784 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_56/ResNet18_32x32_e50_i0.pth i=785 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_56/ResNet18_32x32_e20_i0.pth i=786 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_56/ResNet18_32x32_e10_i0.pth i=787 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_56/ResNet18_32x32_e5_i0.pth i=788 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_56/ResNet18_32x32_e2_i0.pth i=789 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_56/ResNet18_32x32_e1_i0.pth i=790 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_55/ResNet18_32x32_e300_i0.pth i=791 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_55/ResNet18_32x32_e200_i0.pth i=792 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_55/ResNet18_32x32_e100_i0.pth i=793 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_55/ResNet18_32x32_e50_i0.pth i=794 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_55/ResNet18_32x32_e20_i0.pth i=795 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_55/ResNet18_32x32_e10_i0.pth i=796 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_55/ResNet18_32x32_e5_i0.pth i=797 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_55/ResNet18_32x32_e2_i0.pth i=798 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_55/ResNet18_32x32_e1_i0.pth i=799 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_42/ResNet18_32x32_e300_i0.pth i=800 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_42/ResNet18_32x32_e200_i0.pth i=801 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_42/ResNet18_32x32_e100_i0.pth i=802 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_42/ResNet18_32x32_e50_i0.pth i=803 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_42/ResNet18_32x32_e20_i0.pth i=804 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_42/ResNet18_32x32_e10_i0.pth i=805 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_42/ResNet18_32x32_e5_i0.pth i=806 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_42/ResNet18_32x32_e2_i0.pth i=807 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_42/ResNet18_32x32_e1_i0.pth i=808 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_40/ResNet18_32x32_e300_i0.pth i=809 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_40/ResNet18_32x32_e200_i0.pth i=810 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_40/ResNet18_32x32_e100_i0.pth i=811 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_40/ResNet18_32x32_e50_i0.pth i=812 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_40/ResNet18_32x32_e20_i0.pth i=813 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_40/ResNet18_32x32_e10_i0.pth i=814 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_40/ResNet18_32x32_e5_i0.pth i=815 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_40/ResNet18_32x32_e2_i0.pth i=816 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_40/ResNet18_32x32_e1_i0.pth i=817 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_06_00/ResNet18_32x32_e300_i0.pth i=818 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_06_00/ResNet18_32x32_e200_i0.pth i=819 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_06_00/ResNet18_32x32_e100_i0.pth i=820 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_06_00/ResNet18_32x32_e50_i0.pth i=821 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_06_00/ResNet18_32x32_e20_i0.pth i=822 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_06_00/ResNet18_32x32_e10_i0.pth i=823 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_06_00/ResNet18_32x32_e5_i0.pth i=824 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_06_00/ResNet18_32x32_e2_i0.pth i=825 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_06_00/ResNet18_32x32_e1_i0.pth i=826 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_04_07/ResNet18_32x32_e300_i0.pth i=827 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_04_07/ResNet18_32x32_e200_i0.pth i=828 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_04_07/ResNet18_32x32_e100_i0.pth i=829 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_04_07/ResNet18_32x32_e50_i0.pth i=830 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_04_07/ResNet18_32x32_e20_i0.pth i=831 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_04_07/ResNet18_32x32_e10_i0.pth i=832 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_04_07/ResNet18_32x32_e5_i0.pth i=833 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_04_07/ResNet18_32x32_e2_i0.pth i=834 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_04_07/ResNet18_32x32_e1_i0.pth i=835 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_04_06/ResNet18_32x32_e300_i0.pth i=836 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_04_06/ResNet18_32x32_e200_i0.pth i=837 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_04_06/ResNet18_32x32_e100_i0.pth i=838 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_04_06/ResNet18_32x32_e50_i0.pth i=839 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_04_06/ResNet18_32x32_e20_i0.pth i=840 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_04_06/ResNet18_32x32_e10_i0.pth i=841 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_04_06/ResNet18_32x32_e5_i0.pth i=842 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_04_06/ResNet18_32x32_e2_i0.pth i=843 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_04_06/ResNet18_32x32_e1_i0.pth i=844 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_56/ResNet18_32x32_e300_i0.pth i=845 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_56/ResNet18_32x32_e200_i0.pth i=846 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_56/ResNet18_32x32_e100_i0.pth i=847 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_56/ResNet18_32x32_e50_i0.pth i=848 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_56/ResNet18_32x32_e20_i0.pth i=849 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_56/ResNet18_32x32_e10_i0.pth i=850 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_56/ResNet18_32x32_e5_i0.pth i=851 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_56/ResNet18_32x32_e2_i0.pth i=852 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_56/ResNet18_32x32_e1_i0.pth i=853 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_43/ResNet18_32x32_e300_i0.pth i=854 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_43/ResNet18_32x32_e200_i0.pth i=855 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_43/ResNet18_32x32_e100_i0.pth i=856 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_43/ResNet18_32x32_e50_i0.pth i=857 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_43/ResNet18_32x32_e20_i0.pth i=858 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_43/ResNet18_32x32_e10_i0.pth i=859 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_43/ResNet18_32x32_e5_i0.pth i=860 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_43/ResNet18_32x32_e2_i0.pth i=861 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_43/ResNet18_32x32_e1_i0.pth i=862 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_36/ResNet18_32x32_e300_i0.pth i=863 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_36/ResNet18_32x32_e200_i0.pth i=864 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_36/ResNet18_32x32_e100_i0.pth i=865 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_36/ResNet18_32x32_e50_i0.pth i=866 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_36/ResNet18_32x32_e20_i0.pth i=867 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_36/ResNet18_32x32_e10_i0.pth i=868 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_36/ResNet18_32x32_e5_i0.pth i=869 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_36/ResNet18_32x32_e2_i0.pth i=870 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_36/ResNet18_32x32_e1_i0.pth i=871 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e300_i0.pth i=872 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e200_i0.pth i=873 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e100_i0.pth i=874 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e50_i0.pth i=875 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e20_i0.pth i=876 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e10_i0.pth i=877 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e5_i0.pth i=878 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e2_i0.pth i=879 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e1_i0.pth i=880 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_19/ResNet18_32x32_e300_i0.pth i=881 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_19/ResNet18_32x32_e200_i0.pth i=882 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_19/ResNet18_32x32_e100_i0.pth i=883 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_19/ResNet18_32x32_e50_i0.pth i=884 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_19/ResNet18_32x32_e20_i0.pth i=885 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_19/ResNet18_32x32_e10_i0.pth i=886 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_19/ResNet18_32x32_e5_i0.pth i=887 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_19/ResNet18_32x32_e2_i0.pth i=888 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_19/ResNet18_32x32_e1_i0.pth i=889 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_00/ResNet18_32x32_e300_i0.pth i=890 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_00/ResNet18_32x32_e200_i0.pth i=891 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_00/ResNet18_32x32_e100_i0.pth i=892 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_00/ResNet18_32x32_e50_i0.pth i=893 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_00/ResNet18_32x32_e20_i0.pth i=894 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_00/ResNet18_32x32_e10_i0.pth i=895 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_00/ResNet18_32x32_e5_i0.pth i=896 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_00/ResNet18_32x32_e2_i0.pth i=897 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_00/ResNet18_32x32_e1_i0.pth i=898 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_02_39/ResNet18_32x32_e300_i0.pth i=899 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_02_39/ResNet18_32x32_e200_i0.pth i=900 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_02_39/ResNet18_32x32_e100_i0.pth i=901 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_02_39/ResNet18_32x32_e50_i0.pth i=902 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_02_39/ResNet18_32x32_e20_i0.pth i=903 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_02_39/ResNet18_32x32_e10_i0.pth i=904 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_02_39/ResNet18_32x32_e5_i0.pth i=905 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_02_39/ResNet18_32x32_e2_i0.pth i=906 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_02_39/ResNet18_32x32_e1_i0.pth i=907 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_56/ResNet18_32x32_e300_i0.pth i=908 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_56/ResNet18_32x32_e200_i0.pth i=909 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_56/ResNet18_32x32_e100_i0.pth i=910 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_56/ResNet18_32x32_e50_i0.pth i=911 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_56/ResNet18_32x32_e20_i0.pth i=912 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_56/ResNet18_32x32_e10_i0.pth i=913 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_56/ResNet18_32x32_e5_i0.pth i=914 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_56/ResNet18_32x32_e2_i0.pth i=915 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_56/ResNet18_32x32_e1_i0.pth i=916 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_38/ResNet18_32x32_e300_i0.pth i=917 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_38/ResNet18_32x32_e200_i0.pth i=918 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_38/ResNet18_32x32_e100_i0.pth i=919 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_38/ResNet18_32x32_e50_i0.pth i=920 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_38/ResNet18_32x32_e20_i0.pth i=921 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_38/ResNet18_32x32_e10_i0.pth i=922 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_38/ResNet18_32x32_e5_i0.pth i=923 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_38/ResNet18_32x32_e2_i0.pth i=924 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_38/ResNet18_32x32_e1_i0.pth i=925 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_03/ResNet18_32x32_e300_i0.pth i=926 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_03/ResNet18_32x32_e200_i0.pth i=927 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_03/ResNet18_32x32_e100_i0.pth i=928 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_03/ResNet18_32x32_e50_i0.pth i=929 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_03/ResNet18_32x32_e20_i0.pth i=930 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_03/ResNet18_32x32_e10_i0.pth i=931 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_03/ResNet18_32x32_e5_i0.pth i=932 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_03/ResNet18_32x32_e2_i0.pth i=933 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_03/ResNet18_32x32_e1_i0.pth i=934 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_00_43/ResNet18_32x32_e300_i0.pth i=935 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_00_43/ResNet18_32x32_e200_i0.pth i=936 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_00_43/ResNet18_32x32_e100_i0.pth i=937 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_00_43/ResNet18_32x32_e50_i0.pth i=938 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_00_43/ResNet18_32x32_e20_i0.pth i=939 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_00_43/ResNet18_32x32_e10_i0.pth i=940 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_00_43/ResNet18_32x32_e5_i0.pth i=941 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_00_43/ResNet18_32x32_e2_i0.pth i=942 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_00_43/ResNet18_32x32_e1_i0.pth i=943 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_05/ResNet18_32x32_e300_i0.pth i=944 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_05/ResNet18_32x32_e200_i0.pth i=945 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_05/ResNet18_32x32_e100_i0.pth i=946 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_05/ResNet18_32x32_e50_i0.pth i=947 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_05/ResNet18_32x32_e20_i0.pth i=948 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_05/ResNet18_32x32_e10_i0.pth i=949 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_05/ResNet18_32x32_e5_i0.pth i=950 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_05/ResNet18_32x32_e2_i0.pth i=951 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_05/ResNet18_32x32_e1_i0.pth i=952 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_04/ResNet18_32x32_e300_i0.pth i=953 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_04/ResNet18_32x32_e200_i0.pth i=954 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_04/ResNet18_32x32_e100_i0.pth i=955 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_04/ResNet18_32x32_e50_i0.pth i=956 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_04/ResNet18_32x32_e20_i0.pth i=957 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_04/ResNet18_32x32_e10_i0.pth i=958 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_04/ResNet18_32x32_e5_i0.pth i=959 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_04/ResNet18_32x32_e2_i0.pth i=960 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_04/ResNet18_32x32_e1_i0.pth i=961 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_03/ResNet18_32x32_e300_i0.pth i=962 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_03/ResNet18_32x32_e200_i0.pth i=963 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_03/ResNet18_32x32_e100_i0.pth i=964 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_03/ResNet18_32x32_e50_i0.pth i=965 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_03/ResNet18_32x32_e20_i0.pth i=966 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_03/ResNet18_32x32_e10_i0.pth i=967 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_03/ResNet18_32x32_e5_i0.pth i=968 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_03/ResNet18_32x32_e2_i0.pth i=969 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_03/ResNet18_32x32_e1_i0.pth i=970 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_00/ResNet18_32x32_e300_i0.pth i=971 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_00/ResNet18_32x32_e200_i0.pth i=972 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_00/ResNet18_32x32_e100_i0.pth i=973 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_00/ResNet18_32x32_e50_i0.pth i=974 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_00/ResNet18_32x32_e20_i0.pth i=975 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_00/ResNet18_32x32_e10_i0.pth i=976 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_00/ResNet18_32x32_e5_i0.pth i=977 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_00/ResNet18_32x32_e2_i0.pth i=978 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_00/ResNet18_32x32_e1_i0.pth i=979 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_58/ResNet18_32x32_e300_i0.pth i=980 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_58/ResNet18_32x32_e200_i0.pth i=981 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_58/ResNet18_32x32_e100_i0.pth i=982 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_58/ResNet18_32x32_e50_i0.pth i=983 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_58/ResNet18_32x32_e20_i0.pth i=984 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_58/ResNet18_32x32_e10_i0.pth i=985 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_58/ResNet18_32x32_e5_i0.pth i=986 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_58/ResNet18_32x32_e2_i0.pth i=987 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_58/ResNet18_32x32_e1_i0.pth i=988 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_56/ResNet18_32x32_e300_i0.pth i=989 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_56/ResNet18_32x32_e200_i0.pth i=990 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_56/ResNet18_32x32_e100_i0.pth i=991 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_56/ResNet18_32x32_e50_i0.pth i=992 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_56/ResNet18_32x32_e20_i0.pth i=993 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_56/ResNet18_32x32_e10_i0.pth i=994 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_56/ResNet18_32x32_e5_i0.pth i=995 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_56/ResNet18_32x32_e2_i0.pth i=996 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_56/ResNet18_32x32_e1_i0.pth i=997 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_52/ResNet18_32x32_e300_i0.pth i=998 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_52/ResNet18_32x32_e200_i0.pth i=999 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_52/ResNet18_32x32_e100_i0.pth i=1000 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_52/ResNet18_32x32_e50_i0.pth i=1001 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_52/ResNet18_32x32_e20_i0.pth i=1002 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_52/ResNet18_32x32_e10_i0.pth i=1003 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_52/ResNet18_32x32_e5_i0.pth i=1004 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_52/ResNet18_32x32_e2_i0.pth i=1005 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_52/ResNet18_32x32_e1_i0.pth i=1006 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_37/ResNet18_32x32_e300_i0.pth i=1007 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_37/ResNet18_32x32_e200_i0.pth i=1008 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_37/ResNet18_32x32_e100_i0.pth i=1009 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_37/ResNet18_32x32_e50_i0.pth i=1010 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_37/ResNet18_32x32_e20_i0.pth i=1011 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_37/ResNet18_32x32_e10_i0.pth i=1012 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_37/ResNet18_32x32_e5_i0.pth i=1013 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_37/ResNet18_32x32_e2_i0.pth i=1014 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_37/ResNet18_32x32_e1_i0.pth i=1015 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_34/ResNet18_32x32_e300_i0.pth i=1016 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_34/ResNet18_32x32_e200_i0.pth i=1017 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_34/ResNet18_32x32_e100_i0.pth i=1018 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_34/ResNet18_32x32_e50_i0.pth i=1019 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_34/ResNet18_32x32_e20_i0.pth i=1020 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_34/ResNet18_32x32_e10_i0.pth i=1021 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_34/ResNet18_32x32_e5_i0.pth i=1022 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_34/ResNet18_32x32_e2_i0.pth i=1023 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_34/ResNet18_32x32_e1_i0.pth i=1024 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_27/ResNet18_32x32_e300_i0.pth i=1025 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_27/ResNet18_32x32_e200_i0.pth i=1026 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_27/ResNet18_32x32_e100_i0.pth i=1027 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_27/ResNet18_32x32_e50_i0.pth i=1028 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_27/ResNet18_32x32_e20_i0.pth i=1029 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_27/ResNet18_32x32_e10_i0.pth i=1030 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_27/ResNet18_32x32_e5_i0.pth i=1031 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_27/ResNet18_32x32_e2_i0.pth i=1032 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_27/ResNet18_32x32_e1_i0.pth i=1033 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_22/ResNet18_32x32_e300_i0.pth i=1034 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_22/ResNet18_32x32_e200_i0.pth i=1035 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_22/ResNet18_32x32_e100_i0.pth i=1036 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_22/ResNet18_32x32_e50_i0.pth i=1037 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_22/ResNet18_32x32_e20_i0.pth i=1038 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_22/ResNet18_32x32_e10_i0.pth i=1039 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_22/ResNet18_32x32_e5_i0.pth i=1040 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_22/ResNet18_32x32_e2_i0.pth i=1041 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_22/ResNet18_32x32_e1_i0.pth i=1042 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_09/ResNet18_32x32_e300_i0.pth i=1043 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_09/ResNet18_32x32_e200_i0.pth i=1044 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_09/ResNet18_32x32_e100_i0.pth i=1045 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_09/ResNet18_32x32_e50_i0.pth i=1046 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_09/ResNet18_32x32_e20_i0.pth i=1047 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_09/ResNet18_32x32_e10_i0.pth i=1048 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_09/ResNet18_32x32_e5_i0.pth i=1049 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_09/ResNet18_32x32_e2_i0.pth i=1050 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_09/ResNet18_32x32_e1_i0.pth i=1051 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_03/ResNet18_32x32_e300_i0.pth i=1052 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_03/ResNet18_32x32_e200_i0.pth i=1053 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_03/ResNet18_32x32_e100_i0.pth i=1054 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_03/ResNet18_32x32_e50_i0.pth i=1055 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_03/ResNet18_32x32_e20_i0.pth i=1056 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_03/ResNet18_32x32_e10_i0.pth i=1057 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_03/ResNet18_32x32_e5_i0.pth i=1058 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_03/ResNet18_32x32_e2_i0.pth i=1059 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_03/ResNet18_32x32_e1_i0.pth i=1060 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_02_51/ResNet18_32x32_e300_i0.pth i=1061 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_02_51/ResNet18_32x32_e200_i0.pth i=1062 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_02_51/ResNet18_32x32_e100_i0.pth i=1063 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_02_51/ResNet18_32x32_e50_i0.pth i=1064 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_02_51/ResNet18_32x32_e20_i0.pth i=1065 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_02_51/ResNet18_32x32_e10_i0.pth i=1066 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_02_51/ResNet18_32x32_e5_i0.pth i=1067 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_02_51/ResNet18_32x32_e2_i0.pth i=1068 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_02_51/ResNet18_32x32_e1_i0.pth i=1069 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_01_40/ResNet18_32x32_e300_i0.pth i=1070 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_01_40/ResNet18_32x32_e200_i0.pth i=1071 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_01_40/ResNet18_32x32_e100_i0.pth i=1072 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_01_40/ResNet18_32x32_e50_i0.pth i=1073 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_01_40/ResNet18_32x32_e20_i0.pth i=1074 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_01_40/ResNet18_32x32_e10_i0.pth i=1075 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_01_40/ResNet18_32x32_e5_i0.pth i=1076 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_01_40/ResNet18_32x32_e2_i0.pth i=1077 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_01_40/ResNet18_32x32_e1_i0.pth i=1078 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_40_15/ResNet18_32x32_e300_i0.pth i=1079 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_40_15/ResNet18_32x32_e200_i0.pth i=1080 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_40_15/ResNet18_32x32_e100_i0.pth i=1081 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_40_15/ResNet18_32x32_e50_i0.pth i=1082 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_40_15/ResNet18_32x32_e20_i0.pth i=1083 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_40_15/ResNet18_32x32_e10_i0.pth i=1084 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_40_15/ResNet18_32x32_e5_i0.pth i=1085 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_40_15/ResNet18_32x32_e2_i0.pth i=1086 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_40_15/ResNet18_32x32_e1_i0.pth i=1087 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_20_55/ResNet18_32x32_e300_i0.pth i=1088 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_20_55/ResNet18_32x32_e200_i0.pth i=1089 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_20_55/ResNet18_32x32_e100_i0.pth i=1090 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_20_55/ResNet18_32x32_e50_i0.pth i=1091 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_20_55/ResNet18_32x32_e20_i0.pth i=1092 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_20_55/ResNet18_32x32_e10_i0.pth i=1093 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_20_55/ResNet18_32x32_e5_i0.pth i=1094 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_20_55/ResNet18_32x32_e2_i0.pth i=1095 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_20_55/ResNet18_32x32_e1_i0.pth i=1096 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_20_35/ResNet18_32x32_e300_i0.pth i=1097 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_20_35/ResNet18_32x32_e200_i0.pth i=1098 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_20_35/ResNet18_32x32_e100_i0.pth i=1099 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_20_35/ResNet18_32x32_e50_i0.pth i=1100 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_20_35/ResNet18_32x32_e20_i0.pth i=1101 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_20_35/ResNet18_32x32_e10_i0.pth i=1102 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_20_35/ResNet18_32x32_e5_i0.pth i=1103 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_20_35/ResNet18_32x32_e2_i0.pth i=1104 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_20_35/ResNet18_32x32_e1_i0.pth i=1105 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_51_57/ResNet18_32x32_e300_i0.pth i=1106 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_51_57/ResNet18_32x32_e200_i0.pth i=1107 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_51_57/ResNet18_32x32_e100_i0.pth i=1108 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_51_57/ResNet18_32x32_e50_i0.pth i=1109 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_51_57/ResNet18_32x32_e20_i0.pth i=1110 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_51_57/ResNet18_32x32_e10_i0.pth i=1111 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_51_57/ResNet18_32x32_e5_i0.pth i=1112 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_51_57/ResNet18_32x32_e2_i0.pth i=1113 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_51_57/ResNet18_32x32_e1_i0.pth i=1114 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_49_27/ResNet18_32x32_e300_i0.pth i=1115 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_49_27/ResNet18_32x32_e200_i0.pth i=1116 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_49_27/ResNet18_32x32_e100_i0.pth i=1117 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_49_27/ResNet18_32x32_e50_i0.pth i=1118 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_49_27/ResNet18_32x32_e20_i0.pth i=1119 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_49_27/ResNet18_32x32_e10_i0.pth i=1120 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_49_27/ResNet18_32x32_e5_i0.pth i=1121 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_49_27/ResNet18_32x32_e2_i0.pth i=1122 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_49_27/ResNet18_32x32_e1_i0.pth i=1123 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_39_01/ResNet18_32x32_e300_i0.pth i=1124 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_39_01/ResNet18_32x32_e200_i0.pth i=1125 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_39_01/ResNet18_32x32_e100_i0.pth i=1126 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_39_01/ResNet18_32x32_e50_i0.pth i=1127 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_39_01/ResNet18_32x32_e20_i0.pth i=1128 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_39_01/ResNet18_32x32_e10_i0.pth i=1129 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_39_01/ResNet18_32x32_e5_i0.pth i=1130 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_39_01/ResNet18_32x32_e2_i0.pth i=1131 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_39_01/ResNet18_32x32_e1_i0.pth i=1132 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_19_21/ResNet18_32x32_e300_i0.pth i=1133 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_19_21/ResNet18_32x32_e200_i0.pth i=1134 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_19_21/ResNet18_32x32_e100_i0.pth i=1135 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_19_21/ResNet18_32x32_e50_i0.pth i=1136 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_19_21/ResNet18_32x32_e20_i0.pth i=1137 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_19_21/ResNet18_32x32_e10_i0.pth i=1138 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_19_21/ResNet18_32x32_e5_i0.pth i=1139 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_19_21/ResNet18_32x32_e2_i0.pth i=1140 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_19_21/ResNet18_32x32_e1_i0.pth i=1141 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_19_01/ResNet18_32x32_e300_i0.pth i=1142 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_19_01/ResNet18_32x32_e200_i0.pth i=1143 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_19_01/ResNet18_32x32_e100_i0.pth i=1144 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_19_01/ResNet18_32x32_e50_i0.pth i=1145 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_19_01/ResNet18_32x32_e20_i0.pth i=1146 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_19_01/ResNet18_32x32_e10_i0.pth i=1147 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_19_01/ResNet18_32x32_e5_i0.pth i=1148 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_19_01/ResNet18_32x32_e2_i0.pth i=1149 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_19_01/ResNet18_32x32_e1_i0.pth i=1150 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_18_03/ResNet18_32x32_e300_i0.pth i=1151 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_18_03/ResNet18_32x32_e200_i0.pth i=1152 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_18_03/ResNet18_32x32_e100_i0.pth i=1153 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_18_03/ResNet18_32x32_e50_i0.pth i=1154 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_18_03/ResNet18_32x32_e20_i0.pth i=1155 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_18_03/ResNet18_32x32_e10_i0.pth i=1156 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_18_03/ResNet18_32x32_e5_i0.pth i=1157 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_18_03/ResNet18_32x32_e2_i0.pth i=1158 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_18_03/ResNet18_32x32_e1_i0.pth i=1159 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_02_36/ResNet18_32x32_e300_i0.pth i=1160 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_02_36/ResNet18_32x32_e200_i0.pth i=1161 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_02_36/ResNet18_32x32_e100_i0.pth i=1162 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_02_36/ResNet18_32x32_e50_i0.pth i=1163 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_02_36/ResNet18_32x32_e20_i0.pth i=1164 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_02_36/ResNet18_32x32_e10_i0.pth i=1165 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_02_36/ResNet18_32x32_e5_i0.pth i=1166 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_02_36/ResNet18_32x32_e2_i0.pth i=1167 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_02_36/ResNet18_32x32_e1_i0.pth i=1168 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_59_51/ResNet18_32x32_e300_i0.pth i=1169 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_59_51/ResNet18_32x32_e200_i0.pth i=1170 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_59_51/ResNet18_32x32_e100_i0.pth i=1171 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_59_51/ResNet18_32x32_e50_i0.pth i=1172 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_59_51/ResNet18_32x32_e20_i0.pth i=1173 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_59_51/ResNet18_32x32_e10_i0.pth i=1174 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_59_51/ResNet18_32x32_e5_i0.pth i=1175 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_59_51/ResNet18_32x32_e2_i0.pth i=1176 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_59_51/ResNet18_32x32_e1_i0.pth i=1177 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_37_14/ResNet18_32x32_e300_i0.pth i=1178 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_37_14/ResNet18_32x32_e200_i0.pth i=1179 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_37_14/ResNet18_32x32_e100_i0.pth i=1180 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_37_14/ResNet18_32x32_e50_i0.pth i=1181 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_37_14/ResNet18_32x32_e20_i0.pth i=1182 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_37_14/ResNet18_32x32_e10_i0.pth i=1183 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_37_14/ResNet18_32x32_e5_i0.pth i=1184 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_37_14/ResNet18_32x32_e2_i0.pth i=1185 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_37_14/ResNet18_32x32_e1_i0.pth i=1186 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_29_02/ResNet18_32x32_e300_i0.pth i=1187 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_29_02/ResNet18_32x32_e200_i0.pth i=1188 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_29_02/ResNet18_32x32_e100_i0.pth i=1189 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_29_02/ResNet18_32x32_e50_i0.pth i=1190 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_29_02/ResNet18_32x32_e20_i0.pth i=1191 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_29_02/ResNet18_32x32_e10_i0.pth i=1192 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_29_02/ResNet18_32x32_e5_i0.pth i=1193 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_29_02/ResNet18_32x32_e2_i0.pth i=1194 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_29_02/ResNet18_32x32_e1_i0.pth i=1195 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_17_25/ResNet18_32x32_e300_i0.pth i=1196 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_17_25/ResNet18_32x32_e200_i0.pth i=1197 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_17_25/ResNet18_32x32_e100_i0.pth i=1198 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_17_25/ResNet18_32x32_e50_i0.pth i=1199 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_17_25/ResNet18_32x32_e20_i0.pth i=1200 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_17_25/ResNet18_32x32_e10_i0.pth i=1201 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_17_25/ResNet18_32x32_e5_i0.pth i=1202 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_17_25/ResNet18_32x32_e2_i0.pth i=1203 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_17_25/ResNet18_32x32_e1_i0.pth i=1204 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_17_17/ResNet18_32x32_e300_i0.pth i=1205 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_17_17/ResNet18_32x32_e200_i0.pth i=1206 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_17_17/ResNet18_32x32_e100_i0.pth i=1207 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_17_17/ResNet18_32x32_e50_i0.pth i=1208 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_17_17/ResNet18_32x32_e20_i0.pth i=1209 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_17_17/ResNet18_32x32_e10_i0.pth i=1210 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_17_17/ResNet18_32x32_e5_i0.pth i=1211 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_17_17/ResNet18_32x32_e2_i0.pth i=1212 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_17_17/ResNet18_32x32_e1_i0.pth i=1213 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_10_14/ResNet18_32x32_e300_i0.pth i=1214 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_10_14/ResNet18_32x32_e200_i0.pth i=1215 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_10_14/ResNet18_32x32_e100_i0.pth i=1216 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_10_14/ResNet18_32x32_e50_i0.pth i=1217 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_10_14/ResNet18_32x32_e20_i0.pth i=1218 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_10_14/ResNet18_32x32_e10_i0.pth i=1219 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_10_14/ResNet18_32x32_e5_i0.pth i=1220 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_10_14/ResNet18_32x32_e2_i0.pth i=1221 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_10_14/ResNet18_32x32_e1_i0.pth i=1222 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_08_48/ResNet18_32x32_e300_i0.pth i=1223 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_08_48/ResNet18_32x32_e200_i0.pth i=1224 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_08_48/ResNet18_32x32_e100_i0.pth i=1225 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_08_48/ResNet18_32x32_e50_i0.pth i=1226 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_08_48/ResNet18_32x32_e20_i0.pth i=1227 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_08_48/ResNet18_32x32_e10_i0.pth i=1228 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_08_48/ResNet18_32x32_e5_i0.pth i=1229 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_08_48/ResNet18_32x32_e2_i0.pth i=1230 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_08_48/ResNet18_32x32_e1_i0.pth i=1231 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_06_32/ResNet18_32x32_e300_i0.pth i=1232 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_06_32/ResNet18_32x32_e200_i0.pth i=1233 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_06_32/ResNet18_32x32_e100_i0.pth i=1234 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_06_32/ResNet18_32x32_e50_i0.pth i=1235 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_06_32/ResNet18_32x32_e20_i0.pth i=1236 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_06_32/ResNet18_32x32_e10_i0.pth i=1237 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_06_32/ResNet18_32x32_e5_i0.pth i=1238 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_06_32/ResNet18_32x32_e2_i0.pth i=1239 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_06_32/ResNet18_32x32_e1_i0.pth i=1240 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_39_03/ResNet18_32x32_e300_i0.pth i=1241 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_39_03/ResNet18_32x32_e200_i0.pth i=1242 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_39_03/ResNet18_32x32_e100_i0.pth i=1243 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_39_03/ResNet18_32x32_e50_i0.pth i=1244 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_39_03/ResNet18_32x32_e20_i0.pth i=1245 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_39_03/ResNet18_32x32_e10_i0.pth i=1246 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_39_03/ResNet18_32x32_e5_i0.pth i=1247 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_39_03/ResNet18_32x32_e2_i0.pth i=1248 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_39_03/ResNet18_32x32_e1_i0.pth i=1249 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_35_08/ResNet18_32x32_e300_i0.pth i=1250 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_35_08/ResNet18_32x32_e200_i0.pth i=1251 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_35_08/ResNet18_32x32_e100_i0.pth i=1252 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_35_08/ResNet18_32x32_e50_i0.pth i=1253 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_35_08/ResNet18_32x32_e20_i0.pth i=1254 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_35_08/ResNet18_32x32_e10_i0.pth i=1255 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_35_08/ResNet18_32x32_e5_i0.pth i=1256 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_35_08/ResNet18_32x32_e2_i0.pth i=1257 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_35_08/ResNet18_32x32_e1_i0.pth i=1258 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_18_25/ResNet18_32x32_e300_i0.pth i=1259 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_18_25/ResNet18_32x32_e200_i0.pth i=1260 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_18_25/ResNet18_32x32_e100_i0.pth i=1261 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_18_25/ResNet18_32x32_e50_i0.pth i=1262 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_18_25/ResNet18_32x32_e20_i0.pth i=1263 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_18_25/ResNet18_32x32_e10_i0.pth i=1264 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_18_25/ResNet18_32x32_e5_i0.pth i=1265 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_18_25/ResNet18_32x32_e2_i0.pth i=1266 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_18_25/ResNet18_32x32_e1_i0.pth i=1267 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_16_41/ResNet18_32x32_e300_i0.pth i=1268 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_16_41/ResNet18_32x32_e200_i0.pth i=1269 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_16_41/ResNet18_32x32_e100_i0.pth i=1270 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_16_41/ResNet18_32x32_e50_i0.pth i=1271 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_16_41/ResNet18_32x32_e20_i0.pth i=1272 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_16_41/ResNet18_32x32_e10_i0.pth i=1273 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_16_41/ResNet18_32x32_e5_i0.pth i=1274 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_16_41/ResNet18_32x32_e2_i0.pth i=1275 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_16_41/ResNet18_32x32_e1_i0.pth i=1276 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_14_34/ResNet18_32x32_e300_i0.pth i=1277 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_14_34/ResNet18_32x32_e200_i0.pth i=1278 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_14_34/ResNet18_32x32_e100_i0.pth i=1279 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_14_34/ResNet18_32x32_e50_i0.pth i=1280 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_14_34/ResNet18_32x32_e20_i0.pth i=1281 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_14_34/ResNet18_32x32_e10_i0.pth i=1282 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_14_34/ResNet18_32x32_e5_i0.pth i=1283 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_14_34/ResNet18_32x32_e2_i0.pth i=1284 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_14_34/ResNet18_32x32_e1_i0.pth i=1285 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_14_32/ResNet18_32x32_e300_i0.pth i=1286 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_14_32/ResNet18_32x32_e200_i0.pth i=1287 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_14_32/ResNet18_32x32_e100_i0.pth i=1288 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_14_32/ResNet18_32x32_e50_i0.pth i=1289 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_14_32/ResNet18_32x32_e20_i0.pth i=1290 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_14_32/ResNet18_32x32_e10_i0.pth i=1291 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_14_32/ResNet18_32x32_e5_i0.pth i=1292 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_14_32/ResNet18_32x32_e2_i0.pth i=1293 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_14_32/ResNet18_32x32_e1_i0.pth i=1294 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_05_54/ResNet18_32x32_e300_i0.pth i=1295 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_05_54/ResNet18_32x32_e200_i0.pth i=1296 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_05_54/ResNet18_32x32_e100_i0.pth i=1297 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_05_54/ResNet18_32x32_e50_i0.pth i=1298 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_05_54/ResNet18_32x32_e20_i0.pth i=1299 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_05_54/ResNet18_32x32_e10_i0.pth i=1300 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_05_54/ResNet18_32x32_e5_i0.pth i=1301 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_05_54/ResNet18_32x32_e2_i0.pth i=1302 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_05_54/ResNet18_32x32_e1_i0.pth i=1303 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_57_28/ResNet18_32x32_e300_i0.pth i=1304 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_57_28/ResNet18_32x32_e200_i0.pth i=1305 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_57_28/ResNet18_32x32_e100_i0.pth i=1306 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_57_28/ResNet18_32x32_e50_i0.pth i=1307 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_57_28/ResNet18_32x32_e20_i0.pth i=1308 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_57_28/ResNet18_32x32_e10_i0.pth i=1309 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_57_28/ResNet18_32x32_e5_i0.pth i=1310 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_57_28/ResNet18_32x32_e2_i0.pth i=1311 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_57_28/ResNet18_32x32_e1_i0.pth i=1312 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_47_39/ResNet18_32x32_e300_i0.pth i=1313 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_47_39/ResNet18_32x32_e200_i0.pth i=1314 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_47_39/ResNet18_32x32_e100_i0.pth i=1315 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_47_39/ResNet18_32x32_e50_i0.pth i=1316 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_47_39/ResNet18_32x32_e20_i0.pth i=1317 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_47_39/ResNet18_32x32_e10_i0.pth i=1318 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_47_39/ResNet18_32x32_e5_i0.pth i=1319 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_47_39/ResNet18_32x32_e2_i0.pth i=1320 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_47_39/ResNet18_32x32_e1_i0.pth i=1321 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_33_07/ResNet18_32x32_e300_i0.pth i=1322 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_33_07/ResNet18_32x32_e200_i0.pth i=1323 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_33_07/ResNet18_32x32_e100_i0.pth i=1324 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_33_07/ResNet18_32x32_e50_i0.pth i=1325 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_33_07/ResNet18_32x32_e20_i0.pth i=1326 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_33_07/ResNet18_32x32_e10_i0.pth i=1327 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_33_07/ResNet18_32x32_e5_i0.pth i=1328 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_33_07/ResNet18_32x32_e2_i0.pth i=1329 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_33_07/ResNet18_32x32_e1_i0.pth i=1330 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_26_14/ResNet18_32x32_e300_i0.pth i=1331 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_26_14/ResNet18_32x32_e200_i0.pth i=1332 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_26_14/ResNet18_32x32_e100_i0.pth i=1333 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_26_14/ResNet18_32x32_e50_i0.pth i=1334 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_26_14/ResNet18_32x32_e20_i0.pth i=1335 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_26_14/ResNet18_32x32_e10_i0.pth i=1336 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_26_14/ResNet18_32x32_e5_i0.pth i=1337 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_26_14/ResNet18_32x32_e2_i0.pth i=1338 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_26_14/ResNet18_32x32_e1_i0.pth i=1339 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_26_08/ResNet18_32x32_e300_i0.pth i=1340 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_26_08/ResNet18_32x32_e200_i0.pth i=1341 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_26_08/ResNet18_32x32_e100_i0.pth i=1342 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_26_08/ResNet18_32x32_e50_i0.pth i=1343 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_26_08/ResNet18_32x32_e20_i0.pth i=1344 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_26_08/ResNet18_32x32_e10_i0.pth i=1345 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_26_08/ResNet18_32x32_e5_i0.pth i=1346 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_26_08/ResNet18_32x32_e2_i0.pth i=1347 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_26_08/ResNet18_32x32_e1_i0.pth i=1348 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_24_20/ResNet18_32x32_e300_i0.pth i=1349 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_24_20/ResNet18_32x32_e200_i0.pth i=1350 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_24_20/ResNet18_32x32_e100_i0.pth i=1351 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_24_20/ResNet18_32x32_e50_i0.pth i=1352 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_24_20/ResNet18_32x32_e20_i0.pth i=1353 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_24_20/ResNet18_32x32_e10_i0.pth i=1354 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_24_20/ResNet18_32x32_e5_i0.pth i=1355 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_24_20/ResNet18_32x32_e2_i0.pth i=1356 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_24_20/ResNet18_32x32_e1_i0.pth i=1357 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_14_04/ResNet18_32x32_e300_i0.pth i=1358 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_14_04/ResNet18_32x32_e200_i0.pth i=1359 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_14_04/ResNet18_32x32_e100_i0.pth i=1360 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_14_04/ResNet18_32x32_e50_i0.pth i=1361 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_14_04/ResNet18_32x32_e20_i0.pth i=1362 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_14_04/ResNet18_32x32_e10_i0.pth i=1363 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_14_04/ResNet18_32x32_e5_i0.pth i=1364 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_14_04/ResNet18_32x32_e2_i0.pth i=1365 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_14_04/ResNet18_32x32_e1_i0.pth i=1366 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_12_10/ResNet18_32x32_e300_i0.pth i=1367 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_12_10/ResNet18_32x32_e200_i0.pth i=1368 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_12_10/ResNet18_32x32_e100_i0.pth i=1369 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_12_10/ResNet18_32x32_e50_i0.pth i=1370 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_12_10/ResNet18_32x32_e20_i0.pth i=1371 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_12_10/ResNet18_32x32_e10_i0.pth i=1372 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_12_10/ResNet18_32x32_e5_i0.pth i=1373 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_12_10/ResNet18_32x32_e2_i0.pth i=1374 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_12_10/ResNet18_32x32_e1_i0.pth i=1375 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_12_04/ResNet18_32x32_e300_i0.pth i=1376 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_12_04/ResNet18_32x32_e200_i0.pth i=1377 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_12_04/ResNet18_32x32_e100_i0.pth i=1378 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_12_04/ResNet18_32x32_e50_i0.pth i=1379 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_12_04/ResNet18_32x32_e20_i0.pth i=1380 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_12_04/ResNet18_32x32_e10_i0.pth i=1381 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_12_04/ResNet18_32x32_e5_i0.pth i=1382 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_12_04/ResNet18_32x32_e2_i0.pth i=1383 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_12_04/ResNet18_32x32_e1_i0.pth i=1384 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_05_52/ResNet18_32x32_e300_i0.pth i=1385 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_05_52/ResNet18_32x32_e200_i0.pth i=1386 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_05_52/ResNet18_32x32_e100_i0.pth i=1387 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_05_52/ResNet18_32x32_e50_i0.pth i=1388 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_05_52/ResNet18_32x32_e20_i0.pth i=1389 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_05_52/ResNet18_32x32_e10_i0.pth i=1390 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_05_52/ResNet18_32x32_e5_i0.pth i=1391 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_05_52/ResNet18_32x32_e2_i0.pth i=1392 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_05_52/ResNet18_32x32_e1_i0.pth i=1393 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_57_18/ResNet18_32x32_e300_i0.pth i=1394 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_57_18/ResNet18_32x32_e200_i0.pth i=1395 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_57_18/ResNet18_32x32_e100_i0.pth i=1396 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_57_18/ResNet18_32x32_e50_i0.pth i=1397 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_57_18/ResNet18_32x32_e20_i0.pth i=1398 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_57_18/ResNet18_32x32_e10_i0.pth i=1399 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_57_18/ResNet18_32x32_e5_i0.pth i=1400 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_57_18/ResNet18_32x32_e2_i0.pth i=1401 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_57_18/ResNet18_32x32_e1_i0.pth i=1402 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_36_52/ResNet18_32x32_e300_i0.pth i=1403 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_36_52/ResNet18_32x32_e200_i0.pth i=1404 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_36_52/ResNet18_32x32_e100_i0.pth i=1405 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_36_52/ResNet18_32x32_e50_i0.pth i=1406 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_36_52/ResNet18_32x32_e20_i0.pth i=1407 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_36_52/ResNet18_32x32_e10_i0.pth i=1408 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_36_52/ResNet18_32x32_e5_i0.pth i=1409 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_36_52/ResNet18_32x32_e2_i0.pth i=1410 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_36_52/ResNet18_32x32_e1_i0.pth i=1411 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_34_10/ResNet18_32x32_e300_i0.pth i=1412 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_34_10/ResNet18_32x32_e200_i0.pth i=1413 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_34_10/ResNet18_32x32_e100_i0.pth i=1414 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_34_10/ResNet18_32x32_e50_i0.pth i=1415 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_34_10/ResNet18_32x32_e20_i0.pth i=1416 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_34_10/ResNet18_32x32_e10_i0.pth i=1417 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_34_10/ResNet18_32x32_e5_i0.pth i=1418 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_34_10/ResNet18_32x32_e2_i0.pth i=1419 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_34_10/ResNet18_32x32_e1_i0.pth i=1420 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_34_05/ResNet18_32x32_e300_i0.pth i=1421 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_34_05/ResNet18_32x32_e200_i0.pth i=1422 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_34_05/ResNet18_32x32_e100_i0.pth i=1423 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_34_05/ResNet18_32x32_e50_i0.pth i=1424 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_34_05/ResNet18_32x32_e20_i0.pth i=1425 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_34_05/ResNet18_32x32_e10_i0.pth i=1426 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_34_05/ResNet18_32x32_e5_i0.pth i=1427 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_34_05/ResNet18_32x32_e2_i0.pth i=1428 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_34_05/ResNet18_32x32_e1_i0.pth i=1429 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_29_59/ResNet18_32x32_e300_i0.pth i=1430 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_29_59/ResNet18_32x32_e200_i0.pth i=1431 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_29_59/ResNet18_32x32_e100_i0.pth i=1432 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_29_59/ResNet18_32x32_e50_i0.pth i=1433 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_29_59/ResNet18_32x32_e20_i0.pth i=1434 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_29_59/ResNet18_32x32_e10_i0.pth i=1435 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_29_59/ResNet18_32x32_e5_i0.pth i=1436 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_29_59/ResNet18_32x32_e2_i0.pth i=1437 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_29_59/ResNet18_32x32_e1_i0.pth i=1438 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_21_42/ResNet18_32x32_e300_i0.pth i=1439 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_21_42/ResNet18_32x32_e200_i0.pth i=1440 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_21_42/ResNet18_32x32_e100_i0.pth i=1441 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_21_42/ResNet18_32x32_e50_i0.pth i=1442 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_21_42/ResNet18_32x32_e20_i0.pth i=1443 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_21_42/ResNet18_32x32_e10_i0.pth i=1444 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_21_42/ResNet18_32x32_e5_i0.pth i=1445 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_21_42/ResNet18_32x32_e2_i0.pth i=1446 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_21_42/ResNet18_32x32_e1_i0.pth i=1447 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_14_20/ResNet18_32x32_e300_i0.pth i=1448 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_14_20/ResNet18_32x32_e200_i0.pth i=1449 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_14_20/ResNet18_32x32_e100_i0.pth i=1450 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_14_20/ResNet18_32x32_e50_i0.pth i=1451 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_14_20/ResNet18_32x32_e20_i0.pth i=1452 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_14_20/ResNet18_32x32_e10_i0.pth i=1453 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_14_20/ResNet18_32x32_e5_i0.pth i=1454 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_14_20/ResNet18_32x32_e2_i0.pth i=1455 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_14_20/ResNet18_32x32_e1_i0.pth i=1456 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_40/ResNet18_32x32_e300_i0.pth i=1457 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_40/ResNet18_32x32_e200_i0.pth i=1458 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_40/ResNet18_32x32_e100_i0.pth i=1459 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_40/ResNet18_32x32_e50_i0.pth i=1460 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_40/ResNet18_32x32_e20_i0.pth i=1461 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_40/ResNet18_32x32_e10_i0.pth i=1462 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_40/ResNet18_32x32_e5_i0.pth i=1463 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_40/ResNet18_32x32_e2_i0.pth i=1464 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_40/ResNet18_32x32_e1_i0.pth i=1465 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_31/ResNet18_32x32_e300_i0.pth i=1466 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_31/ResNet18_32x32_e200_i0.pth i=1467 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_31/ResNet18_32x32_e100_i0.pth i=1468 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_31/ResNet18_32x32_e50_i0.pth i=1469 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_31/ResNet18_32x32_e20_i0.pth i=1470 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_31/ResNet18_32x32_e10_i0.pth i=1471 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_31/ResNet18_32x32_e5_i0.pth i=1472 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_31/ResNet18_32x32_e2_i0.pth i=1473 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_31/ResNet18_32x32_e1_i0.pth i=1474 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e300_i0.pth i=1475 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e200_i0.pth i=1476 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e100_i0.pth i=1477 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e50_i0.pth i=1478 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e20_i0.pth i=1479 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e10_i0.pth i=1480 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e5_i0.pth i=1481 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e2_i0.pth i=1482 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e1_i0.pth i=1483 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e300_i0.pth i=1484 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e200_i0.pth i=1485 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e100_i0.pth i=1486 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e50_i0.pth i=1487 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e20_i0.pth i=1488 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e10_i0.pth i=1489 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e5_i0.pth i=1490 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e2_i0.pth i=1491 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e1_i0.pth i=1492 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e300_i0.pth i=1493 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e200_i0.pth i=1494 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e100_i0.pth i=1495 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e50_i0.pth i=1496 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e20_i0.pth i=1497 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e10_i0.pth i=1498 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e5_i0.pth i=1499 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e2_i0.pth i=1500 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e1_i0.pth i=1501 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e300_i0.pth i=1502 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e200_i0.pth i=1503 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e100_i0.pth i=1504 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e50_i0.pth i=1505 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e20_i0.pth i=1506 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e10_i0.pth i=1507 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e5_i0.pth i=1508 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e2_i0.pth i=1509 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e1_i0.pth i=1510 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e300_i0.pth i=1511 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e200_i0.pth i=1512 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e100_i0.pth i=1513 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e50_i0.pth i=1514 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e20_i0.pth i=1515 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e10_i0.pth i=1516 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e5_i0.pth i=1517 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e2_i0.pth i=1518 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e1_i0.pth i=1519 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e300_i0.pth i=1520 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e200_i0.pth i=1521 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e100_i0.pth i=1522 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e50_i0.pth i=1523 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e20_i0.pth i=1524 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e10_i0.pth i=1525 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e5_i0.pth i=1526 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e2_i0.pth i=1527 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e1_i0.pth i=1528 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e300_i0.pth i=1529 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e200_i0.pth i=1530 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e100_i0.pth i=1531 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e50_i0.pth i=1532 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e20_i0.pth i=1533 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e10_i0.pth i=1534 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e5_i0.pth i=1535 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e2_i0.pth i=1536 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e1_i0.pth i=1537 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e300_i0.pth i=1538 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e200_i0.pth i=1539 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e100_i0.pth i=1540 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e50_i0.pth i=1541 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e20_i0.pth i=1542 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e10_i0.pth i=1543 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e5_i0.pth i=1544 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e2_i0.pth i=1545 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e1_i0.pth i=1546 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e300_i0.pth i=1547 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e200_i0.pth i=1548 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e100_i0.pth i=1549 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e50_i0.pth i=1550 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e20_i0.pth i=1551 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e10_i0.pth i=1552 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e5_i0.pth i=1553 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e2_i0.pth i=1554 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e1_i0.pth i=1555 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e300_i0.pth i=1556 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e200_i0.pth i=1557 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e100_i0.pth i=1558 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e50_i0.pth i=1559 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e20_i0.pth i=1560 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e10_i0.pth i=1561 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e5_i0.pth i=1562 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e2_i0.pth i=1563 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e1_i0.pth i=1564 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e300_i0.pth i=1565 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e200_i0.pth i=1566 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e100_i0.pth i=1567 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e50_i0.pth i=1568 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e20_i0.pth i=1569 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e10_i0.pth i=1570 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e5_i0.pth i=1571 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e2_i0.pth i=1572 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e1_i0.pth i=1573 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e300_i0.pth i=1574 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e200_i0.pth i=1575 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e100_i0.pth i=1576 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e50_i0.pth i=1577 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e20_i0.pth i=1578 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e10_i0.pth i=1579 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e5_i0.pth i=1580 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e2_i0.pth i=1581 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e1_i0.pth i=1582 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e300_i0.pth i=1583 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e200_i0.pth i=1584 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e100_i0.pth i=1585 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e50_i0.pth i=1586 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e20_i0.pth i=1587 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e10_i0.pth i=1588 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e5_i0.pth i=1589 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e2_i0.pth i=1590 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e1_i0.pth i=1591 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e300_i0.pth i=1592 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e200_i0.pth i=1593 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e100_i0.pth i=1594 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e50_i0.pth i=1595 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e20_i0.pth i=1596 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e10_i0.pth i=1597 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e5_i0.pth i=1598 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e2_i0.pth i=1599 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e1_i0.pth i=1600 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e300_i0.pth i=1601 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e200_i0.pth i=1602 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e100_i0.pth i=1603 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e50_i0.pth i=1604 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e20_i0.pth i=1605 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e10_i0.pth i=1606 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e5_i0.pth i=1607 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e2_i0.pth i=1608 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e1_i0.pth i=1609 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e300_i0.pth i=1610 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e200_i0.pth i=1611 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e100_i0.pth i=1612 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e50_i0.pth i=1613 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e20_i0.pth i=1614 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e10_i0.pth i=1615 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e5_i0.pth i=1616 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e2_i0.pth i=1617 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e1_i0.pth i=1618 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e300_i0.pth i=1619 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e200_i0.pth i=1620 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e100_i0.pth i=1621 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e50_i0.pth i=1622 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e20_i0.pth i=1623 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e10_i0.pth i=1624 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e5_i0.pth i=1625 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e2_i0.pth i=1626 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e1_i0.pth i=1627 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e300_i0.pth i=1628 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e200_i0.pth i=1629 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e100_i0.pth i=1630 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e50_i0.pth i=1631 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e20_i0.pth i=1632 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e10_i0.pth i=1633 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e5_i0.pth i=1634 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e2_i0.pth i=1635 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e1_i0.pth i=1636 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e300_i0.pth i=1637 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e200_i0.pth i=1638 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e100_i0.pth i=1639 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e50_i0.pth i=1640 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e20_i0.pth i=1641 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e10_i0.pth i=1642 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e5_i0.pth i=1643 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e2_i0.pth i=1644 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e1_i0.pth i=1645 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e300_i0.pth i=1646 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e200_i0.pth i=1647 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e100_i0.pth i=1648 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e50_i0.pth i=1649 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e20_i0.pth i=1650 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e10_i0.pth i=1651 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e5_i0.pth i=1652 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e2_i0.pth i=1653 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e1_i0.pth i=1654 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e300_i0.pth i=1655 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e200_i0.pth i=1656 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e100_i0.pth i=1657 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e50_i0.pth i=1658 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e20_i0.pth i=1659 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e10_i0.pth i=1660 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e5_i0.pth i=1661 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e2_i0.pth i=1662 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e1_i0.pth i=1663 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e300_i0.pth i=1664 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e200_i0.pth i=1665 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e100_i0.pth i=1666 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e50_i0.pth i=1667 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e20_i0.pth i=1668 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e10_i0.pth i=1669 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e5_i0.pth i=1670 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e2_i0.pth i=1671 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e1_i0.pth i=1672 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e300_i0.pth i=1673 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e200_i0.pth i=1674 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e100_i0.pth i=1675 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e50_i0.pth i=1676 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e20_i0.pth i=1677 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e10_i0.pth i=1678 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e5_i0.pth i=1679 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e2_i0.pth i=1680 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e1_i0.pth i=1681 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e300_i0.pth i=1682 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e200_i0.pth i=1683 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e100_i0.pth i=1684 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e50_i0.pth i=1685 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e20_i0.pth i=1686 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e10_i0.pth i=1687 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e5_i0.pth i=1688 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e2_i0.pth i=1689 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e1_i0.pth i=1690 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e300_i0.pth i=1691 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e200_i0.pth i=1692 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e100_i0.pth i=1693 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e50_i0.pth i=1694 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e20_i0.pth i=1695 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e10_i0.pth i=1696 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e5_i0.pth i=1697 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e2_i0.pth i=1698 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e1_i0.pth i=1699 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e300_i0.pth i=1700 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e200_i0.pth i=1701 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e100_i0.pth i=1702 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e50_i0.pth i=1703 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e20_i0.pth i=1704 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e10_i0.pth i=1705 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e5_i0.pth i=1706 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e2_i0.pth i=1707 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e1_i0.pth i=1708 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e300_i0.pth i=1709 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e200_i0.pth i=1710 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e100_i0.pth i=1711 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e50_i0.pth i=1712 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e20_i0.pth i=1713 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e10_i0.pth i=1714 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e5_i0.pth i=1715 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e2_i0.pth i=1716 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e1_i0.pth i=1717 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e300_i0.pth i=1718 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e200_i0.pth i=1719 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e100_i0.pth i=1720 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e50_i0.pth i=1721 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e20_i0.pth i=1722 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e10_i0.pth i=1723 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e5_i0.pth i=1724 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e2_i0.pth i=1725 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e1_i0.pth i=1726 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e300_i0.pth i=1727 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e200_i0.pth i=1728 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e100_i0.pth i=1729 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e50_i0.pth i=1730 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e20_i0.pth i=1731 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e10_i0.pth i=1732 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e5_i0.pth i=1733 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e2_i0.pth i=1734 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e1_i0.pth i=1735 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e300_i0.pth i=1736 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e200_i0.pth i=1737 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e100_i0.pth i=1738 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e50_i0.pth i=1739 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e20_i0.pth i=1740 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e10_i0.pth i=1741 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e5_i0.pth i=1742 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e2_i0.pth i=1743 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e1_i0.pth i=1744 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e300_i0.pth i=1745 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e200_i0.pth i=1746 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e100_i0.pth i=1747 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e50_i0.pth i=1748 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e20_i0.pth i=1749 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e10_i0.pth i=1750 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e5_i0.pth i=1751 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e2_i0.pth i=1752 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e1_i0.pth i=1753 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e300_i0.pth i=1754 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e200_i0.pth i=1755 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e100_i0.pth i=1756 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e50_i0.pth i=1757 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e20_i0.pth i=1758 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e10_i0.pth i=1759 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e5_i0.pth i=1760 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e2_i0.pth i=1761 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e1_i0.pth i=1762 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e300_i0.pth i=1763 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e200_i0.pth i=1764 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e100_i0.pth i=1765 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e50_i0.pth i=1766 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e20_i0.pth i=1767 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e10_i0.pth i=1768 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e5_i0.pth i=1769 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e2_i0.pth i=1770 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e1_i0.pth i=1771 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e300_i0.pth i=1772 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e200_i0.pth i=1773 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e100_i0.pth i=1774 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e50_i0.pth i=1775 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e20_i0.pth i=1776 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e10_i0.pth i=1777 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e5_i0.pth i=1778 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e2_i0.pth i=1779 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e1_i0.pth i=1780 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e300_i0.pth i=1781 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e200_i0.pth i=1782 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e100_i0.pth i=1783 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e50_i0.pth i=1784 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e20_i0.pth i=1785 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e10_i0.pth i=1786 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e5_i0.pth i=1787 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e2_i0.pth i=1788 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e1_i0.pth i=1789 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e300_i0.pth i=1790 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e200_i0.pth i=1791 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e100_i0.pth i=1792 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e50_i0.pth i=1793 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e20_i0.pth i=1794 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e10_i0.pth i=1795 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e5_i0.pth i=1796 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e2_i0.pth i=1797 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e1_i0.pth i=1798 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e300_i0.pth i=1799 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e200_i0.pth i=1800 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e100_i0.pth i=1801 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e50_i0.pth i=1802 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e20_i0.pth i=1803 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e10_i0.pth i=1804 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e5_i0.pth i=1805 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e2_i0.pth i=1806 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e1_i0.pth i=1807 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e300_i0.pth i=1808 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e200_i0.pth i=1809 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e100_i0.pth i=1810 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e50_i0.pth i=1811 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e20_i0.pth i=1812 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e10_i0.pth i=1813 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e5_i0.pth i=1814 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e2_i0.pth i=1815 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e1_i0.pth i=1816 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e300_i0.pth i=1817 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e200_i0.pth i=1818 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e100_i0.pth i=1819 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e50_i0.pth i=1820 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e20_i0.pth i=1821 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e10_i0.pth i=1822 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e5_i0.pth i=1823 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e2_i0.pth i=1824 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e1_i0.pth i=1825 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e300_i0.pth i=1826 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e200_i0.pth i=1827 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e100_i0.pth i=1828 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e50_i0.pth i=1829 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e20_i0.pth i=1830 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e10_i0.pth i=1831 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e5_i0.pth i=1832 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e2_i0.pth i=1833 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e1_i0.pth i=1834 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e300_i0.pth i=1835 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e200_i0.pth i=1836 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e100_i0.pth i=1837 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e50_i0.pth i=1838 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e20_i0.pth i=1839 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e10_i0.pth i=1840 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e5_i0.pth i=1841 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e2_i0.pth i=1842 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e1_i0.pth i=1843 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e300_i0.pth i=1844 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e200_i0.pth i=1845 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e100_i0.pth i=1846 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e50_i0.pth i=1847 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e20_i0.pth i=1848 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e10_i0.pth i=1849 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e5_i0.pth i=1850 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e2_i0.pth i=1851 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e1_i0.pth i=1852 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e300_i0.pth i=1853 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e200_i0.pth i=1854 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e100_i0.pth i=1855 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e50_i0.pth i=1856 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e20_i0.pth i=1857 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e10_i0.pth i=1858 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e5_i0.pth i=1859 n=1861' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e2_i0.pth i=1860 n=1861' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute_acc.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e1_i0.pth i=1861 n=1861' &
wait $(jobs -p)


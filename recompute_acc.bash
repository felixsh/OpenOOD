#!/bin/bash

cleanup() {
  pkill -P $$
  wait
  exit
}

trap cleanup SIGINT SIGTERM

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e1_i0.pth i=1 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e2_i0.pth i=2 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e5_i0.pth i=3 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e10_i0.pth i=4 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e20_i0.pth i=5 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e50_i0.pth i=6 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e100_i0.pth i=7 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e200_i0.pth i=8 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e500_i0.pth i=9 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e1000_i0.pth i=10 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e1_i0.pth i=11 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e2_i0.pth i=12 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e5_i0.pth i=13 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e10_i0.pth i=14 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e20_i0.pth i=15 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e50_i0.pth i=16 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e100_i0.pth i=17 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e200_i0.pth i=18 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e500_i0.pth i=19 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e1000_i0.pth i=20 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e1_i0.pth i=21 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e2_i0.pth i=22 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e5_i0.pth i=23 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e10_i0.pth i=24 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e20_i0.pth i=25 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e50_i0.pth i=26 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e100_i0.pth i=27 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e150_i0.pth i=28 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e1_i0.pth i=29 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e2_i0.pth i=30 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e5_i0.pth i=31 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e10_i0.pth i=32 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e20_i0.pth i=33 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e50_i0.pth i=34 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e100_i0.pth i=35 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e150_i0.pth i=36 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e1_i0.pth i=37 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e2_i0.pth i=38 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e5_i0.pth i=39 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e10_i0.pth i=40 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e20_i0.pth i=41 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e50_i0.pth i=42 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e100_i0.pth i=43 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e200_i0.pth i=44 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e1_i0.pth i=45 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e2_i0.pth i=46 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e5_i0.pth i=47 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e10_i0.pth i=48 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e20_i0.pth i=49 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e50_i0.pth i=50 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e100_i0.pth i=51 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e200_i0.pth i=52 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e400_i0.pth i=53 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e1_i0.pth i=54 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e2_i0.pth i=55 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e5_i0.pth i=56 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e10_i0.pth i=57 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e20_i0.pth i=58 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e50_i0.pth i=59 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e100_i0.pth i=60 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e200_i0.pth i=61 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e500_i0.pth i=62 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet/ResNet50/no_noise/150+_epochs/run_e150_2024_11_12-21_40_57/_FabricModule_e1_i0.pth i=63 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet/ResNet50/no_noise/150+_epochs/run_e150_2024_11_12-21_40_57/_FabricModule_e2_i0.pth i=64 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet/ResNet50/no_noise/150+_epochs/run_e150_2024_11_12-21_40_57/_FabricModule_e5_i0.pth i=65 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet/ResNet50/no_noise/150+_epochs/run_e150_2024_11_12-21_40_57/_FabricModule_e10_i0.pth i=66 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet/ResNet50/no_noise/150+_epochs/run_e150_2024_11_12-21_40_57/_FabricModule_e20_i0.pth i=67 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet/ResNet50/no_noise/150+_epochs/run_e150_2024_11_12-21_40_57/_FabricModule_e50_i0.pth i=68 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet/ResNet50/no_noise/150+_epochs/run_e150_2024_11_12-21_40_57/_FabricModule_e100_i0.pth i=69 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet/ResNet50/no_noise/150+_epochs/run_e150_2024_11_12-21_40_57/_FabricModule_e150_i0.pth i=70 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e1_i0.pth i=71 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e2_i0.pth i=72 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e5_i0.pth i=73 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e10_i0.pth i=74 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e20_i0.pth i=75 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e50_i0.pth i=76 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e100_i0.pth i=77 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e200_i0.pth i=78 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e300_i0.pth i=79 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e1_i0.pth i=80 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e2_i0.pth i=81 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e5_i0.pth i=82 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e10_i0.pth i=83 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e20_i0.pth i=84 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e50_i0.pth i=85 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e100_i0.pth i=86 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e200_i0.pth i=87 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e300_i0.pth i=88 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e1_i0.pth i=89 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e2_i0.pth i=90 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e5_i0.pth i=91 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e10_i0.pth i=92 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e20_i0.pth i=93 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e50_i0.pth i=94 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e100_i0.pth i=95 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e200_i0.pth i=96 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e300_i0.pth i=97 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e1_i0.pth i=98 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e2_i0.pth i=99 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e5_i0.pth i=100 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e10_i0.pth i=101 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e20_i0.pth i=102 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e50_i0.pth i=103 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e100_i0.pth i=104 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e200_i0.pth i=105 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e300_i0.pth i=106 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e1_i0.pth i=107 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e2_i0.pth i=108 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e5_i0.pth i=109 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e10_i0.pth i=110 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e20_i0.pth i=111 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e50_i0.pth i=112 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e100_i0.pth i=113 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e200_i0.pth i=114 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e300_i0.pth i=115 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e1_i0.pth i=116 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e2_i0.pth i=117 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e5_i0.pth i=118 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e10_i0.pth i=119 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e20_i0.pth i=120 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e50_i0.pth i=121 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e100_i0.pth i=122 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e200_i0.pth i=123 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e300_i0.pth i=124 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e1_i0.pth i=125 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e2_i0.pth i=126 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e5_i0.pth i=127 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e10_i0.pth i=128 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e20_i0.pth i=129 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e50_i0.pth i=130 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e100_i0.pth i=131 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e200_i0.pth i=132 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e300_i0.pth i=133 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e1_i0.pth i=134 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e2_i0.pth i=135 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e5_i0.pth i=136 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e10_i0.pth i=137 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e20_i0.pth i=138 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e50_i0.pth i=139 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e100_i0.pth i=140 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e200_i0.pth i=141 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e300_i0.pth i=142 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e1_i0.pth i=143 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e2_i0.pth i=144 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e5_i0.pth i=145 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e10_i0.pth i=146 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e20_i0.pth i=147 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e50_i0.pth i=148 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e100_i0.pth i=149 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e200_i0.pth i=150 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e300_i0.pth i=151 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e1_i0.pth i=152 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e2_i0.pth i=153 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e5_i0.pth i=154 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e10_i0.pth i=155 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e20_i0.pth i=156 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e50_i0.pth i=157 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e100_i0.pth i=158 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e200_i0.pth i=159 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e300_i0.pth i=160 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e1_i0.pth i=161 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e2_i0.pth i=162 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e5_i0.pth i=163 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e10_i0.pth i=164 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e20_i0.pth i=165 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e50_i0.pth i=166 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e100_i0.pth i=167 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e200_i0.pth i=168 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e300_i0.pth i=169 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e1_i0.pth i=170 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e2_i0.pth i=171 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e5_i0.pth i=172 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e10_i0.pth i=173 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e20_i0.pth i=174 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e50_i0.pth i=175 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e100_i0.pth i=176 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e200_i0.pth i=177 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e300_i0.pth i=178 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e1_i0.pth i=179 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e2_i0.pth i=180 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e5_i0.pth i=181 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e10_i0.pth i=182 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e20_i0.pth i=183 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e50_i0.pth i=184 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e100_i0.pth i=185 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e200_i0.pth i=186 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e300_i0.pth i=187 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e1_i0.pth i=188 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e2_i0.pth i=189 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e5_i0.pth i=190 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e10_i0.pth i=191 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e20_i0.pth i=192 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e50_i0.pth i=193 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e100_i0.pth i=194 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e200_i0.pth i=195 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e300_i0.pth i=196 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e1_i0.pth i=197 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e2_i0.pth i=198 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e5_i0.pth i=199 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e10_i0.pth i=200 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e20_i0.pth i=201 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e50_i0.pth i=202 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e100_i0.pth i=203 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e200_i0.pth i=204 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e300_i0.pth i=205 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e1_i0.pth i=206 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e2_i0.pth i=207 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e5_i0.pth i=208 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e10_i0.pth i=209 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e20_i0.pth i=210 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e50_i0.pth i=211 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e100_i0.pth i=212 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e200_i0.pth i=213 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e300_i0.pth i=214 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e1_i0.pth i=215 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e2_i0.pth i=216 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e5_i0.pth i=217 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e10_i0.pth i=218 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e20_i0.pth i=219 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e50_i0.pth i=220 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e100_i0.pth i=221 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e200_i0.pth i=222 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e300_i0.pth i=223 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e1_i0.pth i=224 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e2_i0.pth i=225 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e5_i0.pth i=226 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e10_i0.pth i=227 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e20_i0.pth i=228 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e50_i0.pth i=229 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e100_i0.pth i=230 n=232' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e200_i0.pth i=231 n=232' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python compute_acc_train.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e300_i0.pth i=232 n=232' &
wait $(jobs -p)


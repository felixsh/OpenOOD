#!/bin/bash

cleanup() {
  pkill -P $$
  wait
  exit
}

trap cleanup SIGINT SIGTERM

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e500_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e1000_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e500_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e1000_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e150_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e2_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e100_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e150_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e10_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e1_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e400_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e2_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e100_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e500_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet/ResNet50/no_noise/150+_epochs/run_e150_2024_11_12-21_40_57/_FabricModule_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet/ResNet50/no_noise/150+_epochs/run_e150_2024_11_12-21_40_57/_FabricModule_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet/ResNet50/no_noise/150+_epochs/run_e150_2024_11_12-21_40_57/_FabricModule_e5_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet/ResNet50/no_noise/150+_epochs/run_e150_2024_11_12-21_40_57/_FabricModule_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet/ResNet50/no_noise/150+_epochs/run_e150_2024_11_12-21_40_57/_FabricModule_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet/ResNet50/no_noise/150+_epochs/run_e150_2024_11_12-21_40_57/_FabricModule_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet/ResNet50/no_noise/150+_epochs/run_e150_2024_11_12-21_40_57/_FabricModule_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet/ResNet50/no_noise/150+_epochs/run_e150_2024_11_12-21_40_57/_FabricModule_e150_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e1_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e2_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e100_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e5_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e200_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e10_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e1_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e2_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e100_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e5_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e200_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e10_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e1_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e2_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e100_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e5_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e200_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e10_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e1_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e2_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e100_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e5_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e200_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e10_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e1_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e2_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e100_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e5_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e200_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e10_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e1_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e2_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e100_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e5_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e200_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e10_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e1_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e2_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e100_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e5_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e200_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_31/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_31/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_31/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_31/ResNet18_32x32_e10_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_31/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_31/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_31/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_31/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_31/ResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_40/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_40/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_40/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_40/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_40/ResNet18_32x32_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_40/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_40/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_40/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_40/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_14_20/ResNet18_32x32_e1_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_14_20/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_14_20/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_14_20/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_14_20/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_14_20/ResNet18_32x32_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_14_20/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_14_20/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_14_20/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_21_42/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_21_42/ResNet18_32x32_e2_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_21_42/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_21_42/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_21_42/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_21_42/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_21_42/ResNet18_32x32_e100_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_21_42/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_21_42/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_29_59/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_29_59/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_29_59/ResNet18_32x32_e5_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_29_59/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_29_59/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_29_59/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_29_59/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_29_59/ResNet18_32x32_e200_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_29_59/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_34_05/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_34_05/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_34_05/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_34_05/ResNet18_32x32_e10_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_34_05/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_34_05/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_34_05/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_34_05/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_34_05/ResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_34_10/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_34_10/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_34_10/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_34_10/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_34_10/ResNet18_32x32_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_34_10/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_34_10/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_34_10/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_34_10/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_36_52/ResNet18_32x32_e1_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_36_52/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_36_52/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_36_52/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_36_52/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_36_52/ResNet18_32x32_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_36_52/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_36_52/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_36_52/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_57_18/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_57_18/ResNet18_32x32_e2_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_57_18/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_57_18/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_57_18/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_57_18/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_57_18/ResNet18_32x32_e100_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_57_18/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_57_18/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_05_52/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_05_52/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_05_52/ResNet18_32x32_e5_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_05_52/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_05_52/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_05_52/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_05_52/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_05_52/ResNet18_32x32_e200_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_05_52/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_12_04/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_12_04/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_12_04/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_12_04/ResNet18_32x32_e10_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_12_04/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_12_04/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_12_04/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_12_04/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_12_04/ResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_12_10/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_12_10/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_12_10/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_12_10/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_12_10/ResNet18_32x32_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_12_10/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_12_10/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_12_10/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_12_10/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_14_04/ResNet18_32x32_e1_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_14_04/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_14_04/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_14_04/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_14_04/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_14_04/ResNet18_32x32_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_14_04/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_14_04/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_14_04/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_24_20/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_24_20/ResNet18_32x32_e2_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_24_20/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_24_20/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_24_20/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_24_20/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_24_20/ResNet18_32x32_e100_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_24_20/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_24_20/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_26_08/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_26_08/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_26_08/ResNet18_32x32_e5_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_26_08/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_26_08/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_26_08/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_26_08/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_26_08/ResNet18_32x32_e200_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_26_08/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_26_14/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_26_14/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_26_14/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_26_14/ResNet18_32x32_e10_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_26_14/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_26_14/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_26_14/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_26_14/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_26_14/ResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_33_07/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_33_07/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_33_07/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_33_07/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_33_07/ResNet18_32x32_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_33_07/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_33_07/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_33_07/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_33_07/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_47_39/ResNet18_32x32_e1_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_47_39/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_47_39/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_47_39/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_47_39/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_47_39/ResNet18_32x32_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_47_39/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_47_39/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_47_39/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_57_28/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_57_28/ResNet18_32x32_e2_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_57_28/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_57_28/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_57_28/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_57_28/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_57_28/ResNet18_32x32_e100_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_57_28/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-09_57_28/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_05_54/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_05_54/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_05_54/ResNet18_32x32_e5_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_05_54/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_05_54/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_05_54/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_05_54/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_05_54/ResNet18_32x32_e200_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_05_54/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_14_32/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_14_32/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_14_32/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_14_32/ResNet18_32x32_e10_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_14_32/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_14_32/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_14_32/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_14_32/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_14_32/ResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_14_34/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_14_34/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_14_34/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_14_34/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_14_34/ResNet18_32x32_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_14_34/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_14_34/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_14_34/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_14_34/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_16_41/ResNet18_32x32_e1_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_16_41/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_16_41/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_16_41/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_16_41/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_16_41/ResNet18_32x32_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_16_41/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_16_41/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_16_41/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_18_25/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_18_25/ResNet18_32x32_e2_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_18_25/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_18_25/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_18_25/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_18_25/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_18_25/ResNet18_32x32_e100_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_18_25/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_18_25/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_35_08/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_35_08/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_35_08/ResNet18_32x32_e5_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_35_08/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_35_08/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_35_08/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_35_08/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_35_08/ResNet18_32x32_e200_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_35_08/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_39_03/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_39_03/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_39_03/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_39_03/ResNet18_32x32_e10_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_39_03/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_39_03/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_39_03/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_39_03/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-10_39_03/ResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_06_32/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_06_32/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_06_32/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_06_32/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_06_32/ResNet18_32x32_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_06_32/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_06_32/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_06_32/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_06_32/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_08_48/ResNet18_32x32_e1_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_08_48/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_08_48/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_08_48/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_08_48/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_08_48/ResNet18_32x32_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_08_48/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_08_48/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_08_48/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_10_14/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_10_14/ResNet18_32x32_e2_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_10_14/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_10_14/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_10_14/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_10_14/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_10_14/ResNet18_32x32_e100_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_10_14/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_10_14/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_17_17/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_17_17/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_17_17/ResNet18_32x32_e5_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_17_17/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_17_17/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_17_17/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_17_17/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_17_17/ResNet18_32x32_e200_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_17_17/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_17_25/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_17_25/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_17_25/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_17_25/ResNet18_32x32_e10_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_17_25/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_17_25/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_17_25/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_17_25/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_17_25/ResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_29_02/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_29_02/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_29_02/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_29_02/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_29_02/ResNet18_32x32_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_29_02/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_29_02/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_29_02/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_29_02/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_37_14/ResNet18_32x32_e1_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_37_14/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_37_14/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_37_14/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_37_14/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_37_14/ResNet18_32x32_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_37_14/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_37_14/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_37_14/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_59_51/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_59_51/ResNet18_32x32_e2_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_59_51/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_59_51/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_59_51/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_59_51/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_59_51/ResNet18_32x32_e100_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_59_51/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-11_59_51/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_02_36/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_02_36/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_02_36/ResNet18_32x32_e5_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_02_36/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_02_36/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_02_36/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_02_36/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_02_36/ResNet18_32x32_e200_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_02_36/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_18_03/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_18_03/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_18_03/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_18_03/ResNet18_32x32_e10_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_18_03/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_18_03/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_18_03/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_18_03/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_18_03/ResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_19_01/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_19_01/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_19_01/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_19_01/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_19_01/ResNet18_32x32_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_19_01/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_19_01/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_19_01/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_19_01/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_19_21/ResNet18_32x32_e1_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_19_21/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_19_21/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_19_21/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_19_21/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_19_21/ResNet18_32x32_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_19_21/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_19_21/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_19_21/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_39_01/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_39_01/ResNet18_32x32_e2_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_39_01/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_39_01/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_39_01/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_39_01/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_39_01/ResNet18_32x32_e100_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_39_01/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_39_01/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_49_27/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_49_27/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_49_27/ResNet18_32x32_e5_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_49_27/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_49_27/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_49_27/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_49_27/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_49_27/ResNet18_32x32_e200_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_49_27/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_51_57/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_51_57/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_51_57/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_51_57/ResNet18_32x32_e10_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_51_57/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_51_57/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_51_57/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_51_57/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_51_57/ResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_20_35/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_20_35/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_20_35/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_20_35/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_20_35/ResNet18_32x32_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_20_35/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_20_35/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_20_35/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_20_35/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_20_55/ResNet18_32x32_e1_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_20_55/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_20_55/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_20_55/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_20_55/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_20_55/ResNet18_32x32_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_20_55/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_20_55/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_20_55/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_40_15/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_40_15/ResNet18_32x32_e2_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_40_15/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_40_15/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_40_15/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_40_15/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_40_15/ResNet18_32x32_e100_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_40_15/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-13_40_15/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_01_40/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_01_40/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_01_40/ResNet18_32x32_e5_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_01_40/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_01_40/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_01_40/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_01_40/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_01_40/ResNet18_32x32_e200_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_01_40/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_02_51/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_02_51/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_02_51/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_02_51/ResNet18_32x32_e10_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_02_51/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_02_51/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_02_51/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_02_51/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_02_51/ResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_03/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_03/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_03/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_03/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_03/ResNet18_32x32_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_03/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_03/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_03/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_03/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_09/ResNet18_32x32_e1_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_09/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_09/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_09/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_09/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_09/ResNet18_32x32_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_09/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_09/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_09/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_22/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_22/ResNet18_32x32_e2_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_22/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_22/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_22/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_22/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_22/ResNet18_32x32_e100_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_22/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_22/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_27/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_27/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_27/ResNet18_32x32_e5_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_27/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_27/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_27/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_27/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_27/ResNet18_32x32_e200_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_27/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_34/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_34/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_34/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_34/ResNet18_32x32_e10_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_34/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_34/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_34/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_34/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_34/ResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_37/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_37/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_37/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_37/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_37/ResNet18_32x32_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_37/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_37/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_37/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_37/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_52/ResNet18_32x32_e1_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_52/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_52/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_52/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_52/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_52/ResNet18_32x32_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_52/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_52/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_52/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_56/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_56/ResNet18_32x32_e2_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_56/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_56/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_56/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_56/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_56/ResNet18_32x32_e100_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_56/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_56/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_58/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_58/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_58/ResNet18_32x32_e5_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_58/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_58/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_58/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_58/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_58/ResNet18_32x32_e200_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_03_58/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_00/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_00/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_00/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_00/ResNet18_32x32_e10_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_00/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_00/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_00/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_00/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_00/ResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_03/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_03/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_03/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_03/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_03/ResNet18_32x32_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_03/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_03/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_03/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_03/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_04/ResNet18_32x32_e1_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_04/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_04/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_04/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_04/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_04/ResNet18_32x32_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_04/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_04/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_04/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_05/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_05/ResNet18_32x32_e2_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_05/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_05/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_05/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_05/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_05/ResNet18_32x32_e100_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_05/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-14_04_05/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_00_43/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_00_43/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_00_43/ResNet18_32x32_e5_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_00_43/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_00_43/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_00_43/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_00_43/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_00_43/ResNet18_32x32_e200_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_00_43/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_03/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_03/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_03/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_03/ResNet18_32x32_e10_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_03/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_03/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_03/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_03/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_03/ResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_38/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_38/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_38/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_38/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_38/ResNet18_32x32_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_38/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_38/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_38/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_38/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_56/ResNet18_32x32_e1_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_56/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_56/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_56/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_56/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_56/ResNet18_32x32_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_56/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_56/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_01_56/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_02_39/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_02_39/ResNet18_32x32_e2_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_02_39/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_02_39/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_02_39/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_02_39/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_02_39/ResNet18_32x32_e100_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_02_39/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_02_39/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_00/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_00/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_00/ResNet18_32x32_e5_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_00/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_00/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_00/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_00/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_00/ResNet18_32x32_e200_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_00/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_19/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_19/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_19/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_19/ResNet18_32x32_e10_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_19/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_19/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_19/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_19/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_19/ResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_29/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_36/ResNet18_32x32_e1_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_36/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_36/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_36/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_36/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_36/ResNet18_32x32_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_36/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_36/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_36/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_43/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_43/ResNet18_32x32_e2_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_43/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_43/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_43/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_43/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_43/ResNet18_32x32_e100_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_43/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_43/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_56/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_56/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_56/ResNet18_32x32_e5_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_56/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_56/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_56/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_56/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_56/ResNet18_32x32_e200_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_03_56/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_04_06/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_04_06/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_04_06/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_04_06/ResNet18_32x32_e10_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_04_06/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_04_06/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_04_06/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_04_06/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_04_06/ResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_04_07/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_04_07/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_04_07/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_04_07/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_04_07/ResNet18_32x32_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_04_07/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_04_07/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_04_07/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_04_07/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_06_00/ResNet18_32x32_e1_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_06_00/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_06_00/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_06_00/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_06_00/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_06_00/ResNet18_32x32_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_06_00/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_06_00/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_06_00/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_40/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_40/ResNet18_32x32_e2_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_40/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_40/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_40/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_40/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_40/ResNet18_32x32_e100_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_40/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_40/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_42/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_42/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_42/ResNet18_32x32_e5_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_42/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_42/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_42/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_42/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_42/ResNet18_32x32_e200_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_42/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_55/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_55/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_55/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_55/ResNet18_32x32_e10_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_55/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_55/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_55/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_55/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_55/ResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_56/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_56/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_56/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_56/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_56/ResNet18_32x32_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_56/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_56/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_56/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_57_56/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_58_40/ResNet18_32x32_e1_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_58_40/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_58_40/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_58_40/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_58_40/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_58_40/ResNet18_32x32_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_58_40/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_58_40/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_58_40/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_58_48/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_58_48/ResNet18_32x32_e2_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_58_48/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_58_48/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_58_48/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_58_48/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_58_48/ResNet18_32x32_e100_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_58_48/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_58_48/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_59_13/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_59_13/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_59_13/ResNet18_32x32_e5_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_59_13/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_59_13/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_59_13/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_59_13/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_59_13/ResNet18_32x32_e200_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-15_59_13/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_07/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_07/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_07/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_07/ResNet18_32x32_e10_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_07/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_07/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_07/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_07/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_07/ResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_16/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_16/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_16/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_16/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_16/ResNet18_32x32_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_16/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_16/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_16/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_16/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_40/ResNet18_32x32_e1_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_40/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_40/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_40/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_40/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_40/ResNet18_32x32_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_40/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_40/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_00_40/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_01_00/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_01_00/ResNet18_32x32_e2_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_01_00/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_01_00/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_01_00/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_01_00/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_01_00/ResNet18_32x32_e100_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_01_00/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_01_00/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_01_32/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_01_32/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_01_32/ResNet18_32x32_e5_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_01_32/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_01_32/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_01_32/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_01_32/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_01_32/ResNet18_32x32_e200_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_01_32/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_03_12/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_03_12/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_03_12/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_03_12/ResNet18_32x32_e10_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_03_12/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_03_12/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_03_12/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_03_12/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_03_12/ResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_03_13/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_03_13/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_03_13/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_03_13/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_03_13/ResNet18_32x32_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_03_13/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_03_13/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_03_13/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_03_13/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_54_26/ResNet18_32x32_e1_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_54_26/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_54_26/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_54_26/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_54_26/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_54_26/ResNet18_32x32_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_54_26/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_54_26/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_54_26/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_54_33/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_54_33/ResNet18_32x32_e2_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_54_33/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_54_33/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_54_33/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_54_33/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_54_33/ResNet18_32x32_e100_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_54_33/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_54_33/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_15/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_15/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_15/ResNet18_32x32_e5_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_15/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_15/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_15/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_15/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_15/ResNet18_32x32_e200_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_15/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_20/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_20/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_20/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_20/ResNet18_32x32_e10_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_20/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_20/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_20/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_20/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_20/ResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_28/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_28/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_28/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_28/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_28/ResNet18_32x32_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_28/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_28/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_28/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_28/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_30/ResNet18_32x32_e1_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_30/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_30/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_30/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_30/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_30/ResNet18_32x32_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_30/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_30/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_30/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_47/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_47/ResNet18_32x32_e2_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_47/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_47/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_47/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_47/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_47/ResNet18_32x32_e100_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_47/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_55_47/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_56_19/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_56_19/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_56_19/ResNet18_32x32_e5_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_56_19/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_56_19/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_56_19/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_56_19/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_56_19/ResNet18_32x32_e200_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_56_19/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_56_54/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_56_54/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_56_54/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_56_54/ResNet18_32x32_e10_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_56_54/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_56_54/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_56_54/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_56_54/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_56_54/ResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_57_22/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_57_22/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_57_22/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_57_22/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_57_22/ResNet18_32x32_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_57_22/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_57_22/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_57_22/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_57_22/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_57_33/ResNet18_32x32_e1_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_57_33/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_57_33/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_57_33/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_57_33/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_57_33/ResNet18_32x32_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_57_33/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_57_33/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_57_33/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_58_05/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_58_05/ResNet18_32x32_e2_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_58_05/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_58_05/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_58_05/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_58_05/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_58_05/ResNet18_32x32_e100_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_58_05/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_58_05/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_58_34/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_58_34/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_58_34/ResNet18_32x32_e5_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_58_34/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_58_34/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_58_34/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_58_34/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_58_34/ResNet18_32x32_e200_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_58_34/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_59_22/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_59_22/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_59_22/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_59_22/ResNet18_32x32_e10_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_59_22/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_59_22/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_59_22/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_59_22/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_59_22/ResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_59_49/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_59_49/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_59_49/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_59_49/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_59_49/ResNet18_32x32_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_59_49/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_59_49/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_59_49/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-16_59_49/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_46_37/ResNet18_32x32_e1_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_46_37/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_46_37/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_46_37/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_46_37/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_46_37/ResNet18_32x32_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_46_37/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_46_37/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_46_37/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_47_08/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_47_08/ResNet18_32x32_e2_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_47_08/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_47_08/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_47_08/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_47_08/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_47_08/ResNet18_32x32_e100_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_47_08/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_47_08/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_47_48/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_47_48/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_47_48/ResNet18_32x32_e5_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_47_48/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_47_48/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_47_48/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_47_48/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_47_48/ResNet18_32x32_e200_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_47_48/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_11/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_11/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_11/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_11/ResNet18_32x32_e10_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_11/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_11/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_11/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_11/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_11/ResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_13/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_13/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_13/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_13/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_13/ResNet18_32x32_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_13/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_13/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_13/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_13/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_44/ResNet18_32x32_e1_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_44/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_44/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_44/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_44/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_44/ResNet18_32x32_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_44/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_44/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_48_44/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_20/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_20/ResNet18_32x32_e2_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_20/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_20/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_20/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_20/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_20/ResNet18_32x32_e100_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_20/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_20/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_36/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_36/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_36/ResNet18_32x32_e5_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_36/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_36/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_36/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_36/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_36/ResNet18_32x32_e200_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_36/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_49/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_49/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_49/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_49/ResNet18_32x32_e10_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_49/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_49/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_49/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_49/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_49/ResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_50/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_50/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_50/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_50/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_50/ResNet18_32x32_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_50/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_50/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_50/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_49_50/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_50_04/ResNet18_32x32_e1_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_50_04/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_50_04/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_50_04/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_50_04/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_50_04/ResNet18_32x32_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_50_04/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_50_04/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_50_04/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_50_50/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_50_50/ResNet18_32x32_e2_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_50_50/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_50_50/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_50_50/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_50_50/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_50_50/ResNet18_32x32_e100_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_50_50/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_50_50/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_15/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_15/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_15/ResNet18_32x32_e5_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_15/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_15/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_15/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_15/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_15/ResNet18_32x32_e200_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_15/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_23/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_23/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_23/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_23/ResNet18_32x32_e10_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_23/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_23/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_23/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_23/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_23/ResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_34/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_34/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_34/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_34/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_34/ResNet18_32x32_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_34/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_34/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_34/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-17_51_34/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_40_10/ResNet18_32x32_e1_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_40_10/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_40_10/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_40_10/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_40_10/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_40_10/ResNet18_32x32_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_40_10/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_40_10/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_40_10/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_40_49/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_40_49/ResNet18_32x32_e2_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_40_49/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_40_49/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_40_49/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_40_49/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_40_49/ResNet18_32x32_e100_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_40_49/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_40_49/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_22/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_22/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_22/ResNet18_32x32_e5_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_22/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_22/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_22/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_22/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_22/ResNet18_32x32_e200_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_22/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_30/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_30/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_30/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_30/ResNet18_32x32_e10_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_30/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_30/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_30/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_30/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_30/ResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_41/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_41/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_41/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_41/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_41/ResNet18_32x32_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_41/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_41/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_41/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_41_41/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_42_37/ResNet18_32x32_e1_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_42_37/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_42_37/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_42_37/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_42_37/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_42_37/ResNet18_32x32_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_42_37/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_42_37/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_42_37/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_42_38/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_42_38/ResNet18_32x32_e2_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_42_38/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_42_38/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_42_38/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_42_38/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_42_38/ResNet18_32x32_e100_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_42_38/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_42_38/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_09/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_09/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_09/ResNet18_32x32_e5_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_09/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_09/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_09/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_09/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_09/ResNet18_32x32_e200_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_09/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_12/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_12/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_12/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_12/ResNet18_32x32_e10_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_12/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_12/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_12/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_12/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_12/ResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_34/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_34/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_34/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_34/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_34/ResNet18_32x32_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_34/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_34/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_34/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_43_34/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_44_09/ResNet18_32x32_e1_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_44_09/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_44_09/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_44_09/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_44_09/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_44_09/ResNet18_32x32_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_44_09/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_44_09/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_44_09/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_09/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_09/ResNet18_32x32_e2_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_09/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_09/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_09/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_09/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_09/ResNet18_32x32_e100_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_09/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_09/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_19/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_19/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_19/ResNet18_32x32_e5_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_19/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_19/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_19/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_19/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_19/ResNet18_32x32_e200_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_19/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_35/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_35/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_35/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_35/ResNet18_32x32_e10_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_35/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_35/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_35/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_35/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_45_35/ResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_46_33/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_46_33/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_46_33/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_46_33/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_46_33/ResNet18_32x32_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_46_33/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_46_33/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_46_33/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-18_46_33/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_33_15/ResNet18_32x32_e1_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_33_15/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_33_15/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_33_15/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_33_15/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_33_15/ResNet18_32x32_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_33_15/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_33_15/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_33_15/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_30/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_30/ResNet18_32x32_e2_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_30/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_30/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_30/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_30/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_30/ResNet18_32x32_e100_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_30/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_30/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_44/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_44/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_44/ResNet18_32x32_e5_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_44/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_44/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_44/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_44/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_44/ResNet18_32x32_e200_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_44/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_46/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_46/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_46/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_46/ResNet18_32x32_e10_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_46/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_46/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_46/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_46/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_46/ResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_54/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_54/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_54/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_54/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_54/ResNet18_32x32_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_54/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_54/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_54/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_35_54/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_15/ResNet18_32x32_e1_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_15/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_15/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_15/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_15/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_15/ResNet18_32x32_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_15/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_15/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_15/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_29/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_29/ResNet18_32x32_e2_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_29/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_29/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_29/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_29/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_29/ResNet18_32x32_e100_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_29/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_29/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_37/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_37/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_37/ResNet18_32x32_e5_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_37/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_37/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_37/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_37/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_37/ResNet18_32x32_e200_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_37/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_56/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_56/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_56/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_56/ResNet18_32x32_e10_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_56/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_56/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_56/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_56/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_36_56/ResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_23/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_23/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_23/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_23/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_23/ResNet18_32x32_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_23/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_23/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_23/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_23/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_43/ResNet18_32x32_e1_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_43/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_43/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_43/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_43/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_43/ResNet18_32x32_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_43/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_43/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_43/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_48/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_48/ResNet18_32x32_e2_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_48/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_48/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_48/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_48/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_48/ResNet18_32x32_e100_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_48/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_38_48/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_39_58/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_39_58/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_39_58/ResNet18_32x32_e5_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_39_58/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_39_58/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_39_58/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_39_58/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_39_58/ResNet18_32x32_e200_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-19_39_58/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_26_42/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_26_42/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_26_42/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_26_42/ResNet18_32x32_e10_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_26_42/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_26_42/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_26_42/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_26_42/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_26_42/ResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_28_40/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_28_40/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_28_40/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_28_40/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_28_40/ResNet18_32x32_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_28_40/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_28_40/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_28_40/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_28_40/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_29_04/ResNet18_32x32_e1_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_29_04/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_29_04/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_29_04/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_29_04/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_29_04/ResNet18_32x32_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_29_04/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_29_04/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_29_04/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_29_36/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_29_36/ResNet18_32x32_e2_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_29_36/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_29_36/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_29_36/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_29_36/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_29_36/ResNet18_32x32_e100_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_29_36/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_29_36/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_15/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_15/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_15/ResNet18_32x32_e5_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_15/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_15/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_15/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_15/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_15/ResNet18_32x32_e200_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_15/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_22/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_22/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_22/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_22/ResNet18_32x32_e10_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_22/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_22/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_22/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_22/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_22/ResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_50/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_50/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_50/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_50/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_50/ResNet18_32x32_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_50/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_50/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_50/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_30_50/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_31_03/ResNet18_32x32_e1_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_31_03/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_31_03/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_31_03/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_31_03/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_31_03/ResNet18_32x32_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_31_03/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_31_03/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_31_03/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_01/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_01/ResNet18_32x32_e2_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_01/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_01/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_01/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_01/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_01/ResNet18_32x32_e100_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_01/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_01/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_15/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_15/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_15/ResNet18_32x32_e5_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_15/ResNet18_32x32_e10_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_15/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_15/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_15/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_15/ResNet18_32x32_e200_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_15/ResNet18_32x32_e300_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_55/ResNet18_32x32_e1_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_55/ResNet18_32x32_e2_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_55/ResNet18_32x32_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_55/ResNet18_32x32_e10_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_55/ResNet18_32x32_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_55/ResNet18_32x32_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_55/ResNet18_32x32_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=3 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_55/ResNet18_32x32_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_55/ResNet18_32x32_e300_i0.pth method=nc' &
wait $(jobs -p)


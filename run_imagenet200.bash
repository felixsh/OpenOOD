#!/bin/bash

cleanup() {
  pkill -P $$
  wait
  exit
}

trap cleanup SIGINT SIGTERM

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e100_i0.pth method=msp' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e100_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e100_i0.pth method=mds' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e100_i0.pth method=react' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e100_i0.pth method=dice' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e100_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e100_i0.pth method=nusa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e100_i0.pth method=vim' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e100_i0.pth method=ncscore' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e100_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e100_i0.pth method=epa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e100_i0.pth method=acc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e150_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e150_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e150_i0.pth method=odin' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e150_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e150_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e150_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e150_i0.pth method=knn' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e150_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e150_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e150_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e150_i0.pth method=neco' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e150_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e150_i0.pth method=acc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e20_i0.pth method=msp' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e20_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e20_i0.pth method=mds' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e20_i0.pth method=react' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e20_i0.pth method=dice' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e20_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e20_i0.pth method=nusa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e20_i0.pth method=vim' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e20_i0.pth method=ncscore' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e20_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e20_i0.pth method=epa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e20_i0.pth method=acc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e50_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e50_i0.pth method=odin' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e50_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e50_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e50_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e50_i0.pth method=knn' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e50_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e50_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e50_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e50_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e50_i0.pth method=acc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e5_i0.pth method=msp' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e5_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e5_i0.pth method=mds' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e5_i0.pth method=react' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e5_i0.pth method=dice' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e5_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e5_i0.pth method=nusa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e5_i0.pth method=vim' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e5_i0.pth method=ncscore' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e5_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e5_i0.pth method=epa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e5_i0.pth method=acc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e100_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e100_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e100_i0.pth method=odin' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e100_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e100_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e100_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e100_i0.pth method=knn' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e100_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e100_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e100_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e100_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e100_i0.pth method=acc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e150_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e150_i0.pth method=msp' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e150_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e150_i0.pth method=mds' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e150_i0.pth method=react' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e150_i0.pth method=dice' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e150_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e150_i0.pth method=nusa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e150_i0.pth method=vim' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e150_i0.pth method=ncscore' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e150_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e150_i0.pth method=epa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e150_i0.pth method=acc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e20_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e20_i0.pth method=odin' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e20_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e20_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e20_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e20_i0.pth method=knn' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e20_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e20_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e20_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e20_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e20_i0.pth method=acc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e50_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e50_i0.pth method=msp' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e50_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e50_i0.pth method=mds' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e50_i0.pth method=react' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e50_i0.pth method=dice' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e50_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e50_i0.pth method=nusa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e50_i0.pth method=vim' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e50_i0.pth method=ncscore' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e50_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e50_i0.pth method=epa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e50_i0.pth method=acc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e5_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e5_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e5_i0.pth method=odin' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e5_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e5_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e5_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e5_i0.pth method=knn' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e5_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e5_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e5_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e5_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e5_i0.pth method=acc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e100_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e100_i0.pth method=msp' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e100_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e100_i0.pth method=mds' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e100_i0.pth method=react' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e100_i0.pth method=dice' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e100_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e100_i0.pth method=nusa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e100_i0.pth method=vim' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e100_i0.pth method=ncscore' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e100_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e100_i0.pth method=epa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e100_i0.pth method=acc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e200_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e200_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e200_i0.pth method=odin' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e200_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e200_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e200_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e200_i0.pth method=knn' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e200_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e200_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e200_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e200_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e200_i0.pth method=acc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e20_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e20_i0.pth method=msp' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e20_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e20_i0.pth method=mds' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e20_i0.pth method=react' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e20_i0.pth method=dice' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e20_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e20_i0.pth method=nusa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e20_i0.pth method=vim' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e20_i0.pth method=ncscore' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e20_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e20_i0.pth method=epa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e20_i0.pth method=acc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e50_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e50_i0.pth method=odin' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e50_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e50_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e50_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e50_i0.pth method=knn' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e50_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e50_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e50_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e50_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e50_i0.pth method=acc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e5_i0.pth method=msp' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e5_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e5_i0.pth method=mds' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e5_i0.pth method=react' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e5_i0.pth method=dice' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e5_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e5_i0.pth method=nusa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e5_i0.pth method=vim' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e5_i0.pth method=ncscore' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e5_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e5_i0.pth method=epa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e5_i0.pth method=acc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e100_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e100_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e100_i0.pth method=odin' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e100_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e100_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e100_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e100_i0.pth method=knn' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e100_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e100_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e100_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e100_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e100_i0.pth method=acc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e200_i0.pth method=msp' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e200_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e200_i0.pth method=mds' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e200_i0.pth method=react' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e200_i0.pth method=dice' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e200_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e200_i0.pth method=nusa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e200_i0.pth method=vim' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e200_i0.pth method=ncscore' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e200_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e200_i0.pth method=epa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e200_i0.pth method=acc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e20_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e20_i0.pth method=odin' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e20_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e20_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e20_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e20_i0.pth method=knn' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e20_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e20_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e20_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e20_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e20_i0.pth method=acc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e400_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e400_i0.pth method=msp' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e400_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e400_i0.pth method=mds' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e400_i0.pth method=react' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e400_i0.pth method=dice' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e400_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e400_i0.pth method=nusa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e400_i0.pth method=vim' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e400_i0.pth method=ncscore' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e400_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e400_i0.pth method=epa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e400_i0.pth method=acc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e50_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e50_i0.pth method=odin' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e50_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e50_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e50_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e50_i0.pth method=knn' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e50_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e50_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e50_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e50_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e50_i0.pth method=acc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e5_i0.pth method=msp' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e5_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e5_i0.pth method=mds' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e5_i0.pth method=react' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e5_i0.pth method=dice' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e5_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e5_i0.pth method=nusa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e5_i0.pth method=vim' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e5_i0.pth method=ncscore' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e5_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e5_i0.pth method=epa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e5_i0.pth method=acc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e100_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e100_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e100_i0.pth method=odin' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e100_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e100_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e100_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e100_i0.pth method=knn' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e100_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e100_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e100_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e100_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e100_i0.pth method=acc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e200_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e200_i0.pth method=msp' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e200_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e200_i0.pth method=mds' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e200_i0.pth method=react' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e200_i0.pth method=dice' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e200_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e200_i0.pth method=nusa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e200_i0.pth method=vim' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e200_i0.pth method=ncscore' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e200_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e200_i0.pth method=epa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e200_i0.pth method=acc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e20_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e20_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e20_i0.pth method=odin' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e20_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e20_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e20_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e20_i0.pth method=knn' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e20_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e20_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e20_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e20_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e20_i0.pth method=acc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e500_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e500_i0.pth method=msp' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e500_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e500_i0.pth method=mds' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e500_i0.pth method=react' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e500_i0.pth method=dice' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e500_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e500_i0.pth method=nusa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e500_i0.pth method=vim' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e500_i0.pth method=ncscore' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e500_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e500_i0.pth method=epa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e500_i0.pth method=acc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e50_i0.pth method=nc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e50_i0.pth method=msp' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e50_i0.pth method=odin' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e50_i0.pth method=mds' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e50_i0.pth method=react' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e50_i0.pth method=dice' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e50_i0.pth method=knn' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e50_i0.pth method=nusa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e50_i0.pth method=vim' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e50_i0.pth method=ncscore' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e50_i0.pth method=epa' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e50_i0.pth method=acc' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e5_i0.pth method=nc' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e5_i0.pth method=msp' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e5_i0.pth method=odin' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e5_i0.pth method=mds' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e5_i0.pth method=react' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e5_i0.pth method=dice' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e5_i0.pth method=knn' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e5_i0.pth method=nusa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e5_i0.pth method=vim' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e5_i0.pth method=ncscore' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e5_i0.pth method=neco' &
krenew -- sh -c 'CUDA_VISIBLE_DEVICES=4 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e5_i0.pth method=epa' &
wait $(jobs -p)

krenew -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e5_i0.pth method=acc' &
wait $(jobs -p)



#!/bin/bash

cleanup() {
  pkill -P $$
  wait
  exit
}

trap cleanup SIGINT SIGTERM

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e300_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e300_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26/NCAlexNet_e300_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e300_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e300_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19/NCAlexNet_e300_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e300_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e300_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40/NCAlexNet_e300_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e300_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e300_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07/NCAlexNet_e300_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e300_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e300_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37/NCAlexNet_e300_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e300_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e300_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02/NCMobileNetV2_e300_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e300_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e300_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32/NCMobileNetV2_e300_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e300_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e300_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23/NCMobileNetV2_e300_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e300_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e300_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45/NCMobileNetV2_e300_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e300_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e300_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03/NCMobileNetV2_e300_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e300_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_48_19/NCResNet18_32x32_e300_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e300_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_58_30/NCResNet18_32x32_e300_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e300_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_18/NCResNet18_32x32_e300_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e300_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_14-23_59_52/NCResNet18_32x32_e300_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e300_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_26/NCResNet18_32x32_e300_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e300_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_00_49/NCResNet18_32x32_e300_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e300_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_17/NCResNet18_32x32_e300_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e300_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_01_56/NCResNet18_32x32_e300_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e300_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e300_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_e300_2024_11_15-00_02_37/NCResNet18_32x32_e300_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-05_11_58/NCVGG16_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_08_56/NCVGG16_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_10_14/NCVGG16_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_11_25/NCVGG16_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e300_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e300_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04/NCVGG16_e300_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e300_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e300_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35/NCVGG16_e300_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e300_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e300_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16/NCVGG16_e300_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e300_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e300_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30/NCVGG16_e300_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e300_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e300_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e300_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e300_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e300_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e300_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e300_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e300_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_17_27/ResNet18_32x32_e300_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e300_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e300_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_18_47/ResNet18_32x32_e300_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e300_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e300_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_28_59/ResNet18_32x32_e300_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e300_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e300_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-06_48_07/ResNet18_32x32_e300_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e300_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e300_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_04_34/ResNet18_32x32_e300_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e300_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e300_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_05_26/ResNet18_32x32_e300_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e300_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e300_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_14_33/ResNet18_32x32_e300_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e300_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e300_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_22_50/ResNet18_32x32_e300_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e300_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e300_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_27_19/ResNet18_32x32_e300_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e300_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e300_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_29_41/ResNet18_32x32_e300_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e300_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e300_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_03/ResNet18_32x32_e300_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e300_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e300_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_43_49/ResNet18_32x32_e300_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e300_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e300_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-07_44_42/ResNet18_32x32_e300_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e300_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e300_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_05_59/ResNet18_32x32_e300_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_31/ResNet18_32x32_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_31/ResNet18_32x32_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_31/ResNet18_32x32_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_31/ResNet18_32x32_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_31/ResNet18_32x32_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_31/ResNet18_32x32_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_40/ResNet18_32x32_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_40/ResNet18_32x32_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_40/ResNet18_32x32_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_40/ResNet18_32x32_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_40/ResNet18_32x32_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_08_40/ResNet18_32x32_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_14_20/ResNet18_32x32_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_14_20/ResNet18_32x32_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_14_20/ResNet18_32x32_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_14_20/ResNet18_32x32_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_14_20/ResNet18_32x32_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_14_20/ResNet18_32x32_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_21_42/ResNet18_32x32_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_21_42/ResNet18_32x32_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_21_42/ResNet18_32x32_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_21_42/ResNet18_32x32_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_21_42/ResNet18_32x32_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-08_21_42/ResNet18_32x32_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e500_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e500_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e500_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e1000_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e1000_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_54_53/ResNet18_32x32_e1000_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e500_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e500_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e500_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e1000_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e1000_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar100/ResNet18_32x32/no_noise/1000+_epochs/run_e1000_2024_11_14-23_55_56/ResNet18_32x32_e1000_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e150_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e150_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-04_15_50/ResNet18_224x224_e150_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e150_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e150_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/150+_epochs/run_e150_2024_11_12-06_49_40/_FabricModule_e150_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/200+_epochs/run_e200_2024_11_14-16_19_59/_FabricModule_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e400_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e400_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/400+_epochs/run_e400_2024_11_14-03_24_57/_FabricModule_e400_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e1_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e1_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e1_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e2_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e2_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e2_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e5_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e5_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e5_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e10_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e10_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e10_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e20_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e20_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e20_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e50_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e50_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e50_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e100_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e100_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e100_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e200_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e200_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e200_i0.pth method=neco' &
wait $(jobs -p)

krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=0 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e500_i0.pth method=vim' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=1 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e500_i0.pth method=epa' &
krenew -t -- sh -c 'CUDA_VISIBLE_DEVICES=2 python recompute.py ckpt=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/ResNet18_224x224/no_noise/500+_epochs/run_e500_2024_11_12-22_24_32/_FabricModule_e500_i0.pth method=neco' &
wait $(jobs -p)


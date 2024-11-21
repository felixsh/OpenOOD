#!/bin/bash

# msp odin mds react dice knn nusa vim ncscore neco epa

cleanup() {
  echo "Terminating all background tasks..."
  CPIDS='pgrep -P $$'
  kill -15 $$
  wait
  for cpid in $CPIDS ; do kill -15 $cpid ; done
  wait
  exit
}

trap cleanup SIGINT SIGTERM

OOD="[msp,odin,mds,react,dice,knn,nusa,vim,ncscore,neco,epa]"
DIR=/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/

CUDA_VISIBLE_DEVICES=0 python eval_main.py run="${DIR}NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_33_02" ood=$OOD &
CUDA_VISIBLE_DEVICES=1 python eval_main.py run="${DIR}NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-05_50_32" ood=$OOD &
CUDA_VISIBLE_DEVICES=2 python eval_main.py run="${DIR}NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_23" ood=$OOD &
CUDA_VISIBLE_DEVICES=3 python eval_main.py run="${DIR}NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_22_45" ood=$OOD &
CUDA_VISIBLE_DEVICES=4 python eval_main.py run="${DIR}NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-05_28_26" ood=$OOD &
CUDA_VISIBLE_DEVICES=5 python eval_main.py run="${DIR}NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_03_19" ood=$OOD &
CUDA_VISIBLE_DEVICES=6 python eval_main.py run="${DIR}NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_06_40" ood=$OOD &
CUDA_VISIBLE_DEVICES=7 python eval_main.py run="${DIR}NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_07" ood=$OOD &
wait $(jobs -p)

CUDA_VISIBLE_DEVICES=0 python eval_main.py run="${DIR}NCAlexNet/no_noise/300+_epochs/run_e300_2024_11_14-06_21_37" ood=$OOD &
CUDA_VISIBLE_DEVICES=1 python eval_main.py run="${DIR}NCMobileNetV2/no_noise/300+_epochs/run_e300_2024_11_14-06_23_03" ood=$OOD &
CUDA_VISIBLE_DEVICES=2 python eval_main.py run="${DIR}NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_13_04" ood=$OOD &
CUDA_VISIBLE_DEVICES=3 python eval_main.py run="${DIR}NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_19_35" ood=$OOD &
CUDA_VISIBLE_DEVICES=4 python eval_main.py run="${DIR}NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_20_16" ood=$OOD &
CUDA_VISIBLE_DEVICES=5 python eval_main.py run="${DIR}NCVGG16/no_noise/300+_epochs/run_e300_2024_11_14-06_24_30" ood=$OOD &
wait $(jobs -p)

trap 'pkill -P $$; exit' SIGINT SIGTERM

BENCHMARK=imagenet200

CUDA_VISIBLE_DEVICES=1 python main_ood.py benchmark=$BENCHMARK postprocessor=msp &
CUDA_VISIBLE_DEVICES=2 python main_ood.py benchmark=$BENCHMARK postprocessor=mds &
wait $(jobs -p)
CUDA_VISIBLE_DEVICES=1 python main_ood.py benchmark=$BENCHMARK postprocessor=odin &
CUDA_VISIBLE_DEVICES=2 python main_ood.py benchmark=$BENCHMARK postprocessor=knn &
wait $(jobs -p)

CUDA_VISIBLE_DEVICES=1 python main_ood.py benchmark=$BENCHMARK postprocessor=react &
CUDA_VISIBLE_DEVICES=2 python main_ood.py benchmark=$BENCHMARK postprocessor=dice &
wait $(jobs -p)
CUDA_VISIBLE_DEVICES=1 python main_ood.py benchmark=$BENCHMARK postprocessor=nusa &
CUDA_VISIBLE_DEVICES=2 python main_ood.py benchmark=$BENCHMARK postprocessor=vim &
wait $(jobs -p)

CUDA_VISIBLE_DEVICES=1 python main_ood.py benchmark=$BENCHMARK postprocessor=ncscore &
CUDA_VISIBLE_DEVICES=2 python main_ood.py benchmark=$BENCHMARK postprocessor=neco &
wait $(jobs -p)
CUDA_VISIBLE_DEVICES=1 python main_ood.py benchmark=$BENCHMARK postprocessor=epa &
wait $(jobs -p)
from eval_main import get_previous_ckpts, get_run_ckpts


devices = [4, 5, 6, 7]

# methods = ['knn', 'nusa', 'vim', 'ncscore', 'neco', 'epa']
methods = ['msp', 'odin', 'mds', 'react', 'dice', 'knn', 'nusa', 'vim', 'ncscore', 'neco', 'epa']


# ckpts = get_previous_ckpts()
run_dir = '/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet/ResNet50/no_noise/150+_epochs/run_e150_2024_11_12-21_40_57'
ckpts = get_run_ckpts(run_dir)


combinations = [(str(c), m) for m in methods for c in ckpts][::-1]


start = '''#!/bin/bash

cleanup() {
  pkill -P $$
  wait
  exit
}

trap cleanup SIGINT SIGTERM\n
'''

template = "krenew -- sh -c 'CUDA_VISIBLE_DEVICES={device} python recompute.py ckpt={ckpt} method={method}' &\n"

delimiter = 'wait $(jobs -p)\n\n'


with open('recompute.bash', 'w') as f:
    f.write(start)

    while combinations:
        for d in devices:
            
            try:
                c, m = combinations.pop()
            except IndexError:
                continue

            f.write(template.format(device=d, ckpt=c, method=m))
        
        f.write(delimiter)

from eval_main import get_previous_ckpts


devices = [0, 1, 2]
methods = ['knn', 'nusa', 'vim', 'ncscore', 'neco', 'epa']
ckpts = get_previous_ckpts()


combinations = [(str(c), m) for m in methods for c in ckpts][::-1]


start = '''
#!/bin/bash

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

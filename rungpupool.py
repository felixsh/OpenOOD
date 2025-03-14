import argparse
from pathlib import Path

import gpuMultiprocessing

"""https://gitlab.com/paloha/gpuMultiprocessing"""


def read_file_lines(file_path: Path) -> list[str]:
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file.readlines()]
    return lines


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run commands in multiprocesing GPU pool'
    )
    parser.add_argument('file_path', type=Path, help='Path to the file with commands')
    parser.add_argument(
        '--devices', type=int, nargs='+', help='List of GPU IDs (integers)'
    )
    args = parser.parse_args()

    command_queue = read_file_lines(args.file_path)
    gpu_id_list = args.devices

    gpuMultiprocessing.queue_runner(
        command_queue,
        gpu_id_list,
        env_gpu_name='CUDA_VISIBLE_DEVICES',
        processes_per_gpu=1,
        allowed_restarts=1,
    )

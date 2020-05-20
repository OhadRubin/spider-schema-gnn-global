import os
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

available_gpus = {}
gpus_usage = defaultdict(int)


def get_next_gpu():
    free_slots = {gpu: available - gpus_usage[gpu] for gpu, available in available_gpus.items()}
    return max(free_slots, key=free_slots.get)


def run(cmd):
    with open(os.devnull, 'wb') as devnull:
        gpu = get_next_gpu()
        gpus_usage[gpu] += 1

        try:
            print(f'Starting command: {cmd}, gpu: {gpu} ')
            cmd_p = ['python'] + cmd.split(' ')

            os.putenv('CUDA_VISIBLE_DEVICES', str(gpu))
            subprocess.check_call(cmd_p, stdout=devnull, stderr=subprocess.STDOUT)
            # os.system(' '.join(cmd))
        except subprocess.CalledProcessError as e:
            print('error: ', e)
        except Exception as e:
            print('error: ', e)
        finally:
            gpus_usage[gpu] -= 1

        print(f'Finished command: {cmd}')


def refresh(executor):
    with open('exec/available_gpus.txt') as f:
        for line in f.readlines():
            gpu, available = line.split(':')
            gpu = int(gpu)
            available = int(available)
            available_gpus[gpu] = available
    executor._max_workers = sum(available_gpus.values())

    queue_file_path = 'exec/queue.txt'
    processed_file_path = 'exec/processed.txt'

    queue_file = open(queue_file_path, 'rt')
    processed_file = open(processed_file_path, 'a')
    for line in queue_file:
        executor.submit(run, line.strip())
        processed_file.write(line.strip() + '\n')
    queue_file.close()
    queue_file = open(queue_file_path, "wt")
    queue_file.close()


if __name__ == '__main__':
    with ThreadPoolExecutor(1) as executor:
        while True:
            refresh(executor)
            time.sleep(10)
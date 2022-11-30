import itertools
import os
import multiprocessing
import pprint
import time

GPUS = [0, 1, 3]
NUM_PROCESS = len(GPUS)

def run_function(loss):
    start = time.time()
    i = multiprocessing.current_process()._identity[0]
    name = "shuffle_" + loss.replace(" ", "_")
    if os.path.exists(f"logs/{name}.log"):
        return
    print(f"python3 run.py -d pano -m shuffle -o {name}.h5 --loss {loss} -g {GPUS[i-1]} -e 40 -lr 0.0001 &> logs/{name}.log")
    os.system(f"python3 run.py -d pano -m shuffle -o {name}.h5 --loss {loss} -g {GPUS[i-1]} -e 40 -lr 0.0001 &> logs/{name}.log")
    print(f"{name} is done: took {time.time() - start}s")
    # time.sleep(0.01)

"""Runs a batch for all combinations of loss functions"""
if __name__ == "__main__":
    losses = ["ssim", "l1", "berhu", "sobel", "edges", "smooth"]
    loss_combinations = []

    for i in range(len(losses)+1):
        loss_combinations += itertools.combinations(losses, i)
    loss_combinations = [l for l in loss_combinations if "ssim" in l] + [(l,) for l in losses]
    loss_combinations = list(set(loss_combinations))
    print(f"Preparing to train {len(loss_combinations)} models")

    loss_combinations = [" ".join(l) for l in loss_combinations]
    pprint.pprint(loss_combinations)

    with multiprocessing.Pool(NUM_PROCESS) as pool:
        pool.map(run_function, loss_combinations)

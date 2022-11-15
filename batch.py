import itertools
import os
import multiprocessing
import pprint
import time

def run_function(loss):
    i = multiprocessing.current_process()._identity[0]
    name = "mobile_" + loss.replace(",", "_") + ".h5"
    print(f"python3 run.py -d pano -m mobile -o {name} --loss {loss} -g {i-1} -e 40 -lr 0.0001")
    time.sleep(1)

if __name__ == "__main__":
    losses = ["ssim", "l1", "berhu", "sobel", "edges", "smooth"]
    loss_combinations = []

    for i in range(len(losses)):
        loss_combinations += itertools.combinations(losses, i)
    loss_combinations = [l for l in loss_combinations if "ssim" in l] + [(l,) for l in losses]
    loss_combinations = list(set(loss_combinations))

    loss_combinations = [",".join(l) for l in loss_combinations]
    pprint.pprint(loss_combinations)
    
    with multiprocessing.Pool(4) as pool:
        pool.map(run_function, loss_combinations)
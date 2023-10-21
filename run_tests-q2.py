#!/usr/bin/env python3
# import os
from subprocess import check_output
import re

# from time import sleep
# import matplotlib
from statistics import mean

# matplotlib.use("Agg")
# from matplotlib import pyplot as plt

#
#  Feel free (a.k.a. you have to) to modify this to instrument your code
#

NUM_SAMPLES = 10
THREADS = [0, 8, 16]
LOOPS = range(1, 801, 8)
inps = ["coarse.txt", "fine.txt"]
lengths = [100, 100000]
iteration_max = 25
csvs = []
for idx, inp in enumerate(inps):
    times = {}
    print(f"starting processing for file {inp}")
    if iteration_max > lengths[idx]:
        step = 1
    else:
        step = lengths[idx] // iteration_max
    hash_workers = [1, 2, 4, 8, 16, lengths[idx]]
    # if step > 1:
    #    hash_workers.append(1)
    # hash_workers.extend(range(step, lengths[idx] + 1, step))
    for dw in range(2):
        for hw in hash_workers:
            if dw == 1:
                dw_num = hw
            else:
                dw_num = 1
            cmd = f"go run src/BST.go -filename=input/{inp} -hash-workers={hw} -data-workers={dw_num} -show-hashtime=false -print-groups=false"
            for i in range(NUM_SAMPLES):
                out = check_output(cmd, shell=True).decode("ascii")
                # print(f"output: {out}")
                m = re.search(r"[-+]?\d*\.\d+e[-+]?\d+|\b\d+\.\d+\b", out)
                # print(f"m: {m}")
                if m:
                    time = float(m.group())
                    if hw not in times:
                        times[hw] = [time]
                    else:
                        times[hw].append(time)
            avg_time = mean(times[hw])
            print(
                f"average hashGroupTime for {hw} hash-workers and {dw_num} data-workers: {avg_time:.4e}"
            )

"""
for thr in THREADS:
    csv = []
    # csv = ["{}/{}".format(inp, loop)]
    for loop in LOOPS:
        cmd = "./bin/prefix_scan -o temp.txt -n {} -i tests/{} -l {}".format(
            thr, inp, loop
        )
        out = check_output(cmd, shell=True).decode("ascii")
        m = re.search("time: (.*)", out)
        if m is not None:
            time = m.group(1)
            csv.append(time)
    csvs.append(csv)
    sleep(0.5)
fig, ax = plt.subplots(figsize=(10, 8))
for idx, csv in enumerate(csvs):
    csv_int = [int(c) for c in csv]
    plt.plot(LOOPS, csv_int, label=THREADS[idx])
plt.xlabel("number of loops")
plt.ylabel("total runtime")
ax.grid()
plt.title("Runtime vs Loops for %s input" % inp)

plt.legend()
out_png = "graph_part2.png"
plt.savefig(out_png, dpi=150)
plt.show()
plt.close()
"""

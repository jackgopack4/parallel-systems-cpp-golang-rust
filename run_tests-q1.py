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
inps = ["coarse,txt", "fine.txt"]
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
    if idx == 0:
        hash_workers = [
            2,
            4,
            8,
            12,
            16,
            20,
            24,
            28,
            32,
            36,
            40,
            44,
            48,
            52,
            56,
            60,
            64,
            68,
            72,
            76,
            80,
            84,
            88,
            92,
            96,
            100,
            1,
        ]
    else:
        hash_workers = [
            2,
            4,
            8,
            16,
            100000,
            4000,
            8000,
            12000,
            16000,
            20000,
            24000,
            28000,
            32000,
            36000,
            40000,
            44000,
            48000,
            52000,
            56000,
            60000,
            64000,
            68000,
            72000,
            76000,
            80000,
            84000,
            88000,
            92000,
            96000,
            1,
        ]
    # if step > 1:
    #    hash_workers.append(1)
    # hash_workers.extend(range(step, lengths[idx] + 1, step))

    for hw in hash_workers:
        cmd = f"go run src/BST.go -filename=input/{inp} -equal-workers=true -hash-workers={hw}"
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
        print(f"{avg_time:.4e}")

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

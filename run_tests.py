#!/usr/bin/env python3
import os
from subprocess import check_output
import re
from time import sleep
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
#
#  Feel free (a.k.a. you have to) to modify this to instrument your code
#

THREADS = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32]
LOOPS = [100000]
INPUTS = ["1k.txt","8k.txt","16k.txt"]

csvs = []

for inp in INPUTS:
    for loop in LOOPS:
        csv = []
        for thr in THREADS:
            cmd = "./bin/prefix_scan -o temp.txt -n {} -i tests/{} -l {}".format(
                thr, inp, loop)
            out = check_output(cmd, shell=True).decode("ascii")
            m = re.search("time: (.*)", out)
            if m is not None:
                time = m.group(1)
                csv.append(time)

        csvs.append(csv)
        sleep(0.5)
fig, ax = plt.subplots(figsize=(10,8))
for idx,csv in enumerate(csvs):
    csv_int = [int(c) for c in csv]
    plt.plot(THREADS,csv_int,label=INPUTS[idx])
    i = 0
plt.xlabel('number of threads')
plt.ylabel('total runtime')
ax.grid()
plt.title("Runtime vs Thread Count for 100,000 Loops")
plt.yticks(range(0,6000000,500000))
plt.xticks([0,4,8,12,16,20,24,28,32])

plt.legend()
out_png = 'graph_part1.png'
plt.savefig(out_png, dpi=150)
plt.show()
plt.close()    

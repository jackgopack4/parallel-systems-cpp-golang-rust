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

THREADS = [0,8,16]
LOOPS = range(1,801,8)
inp = "8k.txt"

csvs = []

for thr in THREADS:
    csv = []
    for loop in LOOPS:
        cmd = "./bin/prefix_scan -o temp.txt -n {} -i tests/{} -l {} -s".format(thr, inp, loop)
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
    plt.plot(LOOPS,csv_int,label=THREADS[idx])

plt.xlabel('number of loops')
plt.ylabel('total runtime')
ax.grid()
plt.title("Runtime vs Loops for %s input with custom barrier" % inp)

plt.legend()
out_png = 'graph_part3.png'
plt.savefig(out_png, dpi=150)
plt.show()
plt.close()
    

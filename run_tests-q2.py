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
LOOPS = range(1,505,8)
inp = "8k.txt"

csvs = []

for thr in THREADS:
    csv = []
    #csv = ["{}/{}".format(inp, loop)]
    for loop in LOOPS:
        cmd = "./bin/prefix_scan -o temp.txt -n {} -i tests/{} -l {}".format(thr, inp, loop)
        out = check_output(cmd, shell=True).decode("ascii")
        m = re.search("time: (.*)", out)
        if m is not None:
            time = m.group(1)
            csv.append(time)
    csvs.append(csv)
    sleep(0.5)
    #print('csv = %s' % csv)
#print(csvs)
#header = ["microseconds"] + [str(x) for x in THREADS]

#csvs.append(['333410', '330277', '172760', '114140', '111973', '135518', '107745', '111553', '105496', '110537', '106928', '103502', '148373', '130576', '130282', '113498', '115281'])
#csvs.append(['2616074', '2674083', '1384607', '941910', '725825', '849241', '850564', '804507', '810442', '943039', '976963', '900839', '762033', '766117', '799899', '1123466', '848675'])
#csvs.append(['5504627', '5593646', '3328401', '2589913', '2069395', '2312344', '1666677', '1496912', '1438729', '1448900', '1441373', '1522026', '1665908', '1571994', '1713533', '1400213', '1456668'])
#print("\n")
#print(", ".join(header))
fig, ax = plt.subplots(figsize=(10,8))
#yticks = []
for idx,csv in enumerate(csvs):
    csv_int = [int(c) for c in csv]
    #print('csv_int = %s'%csv_int)
    #print('loops = %s'%LOOPS)
    plt.plot(LOOPS,csv_int,label=THREADS[idx])
    #i = 0
    #yticks.extend(csv_int)
    #plt.yticks([])
    #print (", ".join(csv))
#yticks.sort()
#yticks = [y for idx, y in enumerate(yticks) if idx % 2 == 0]
#print('yticks = %s' % yticks)
plt.xlabel('number of loops')
plt.ylabel('total runtime')
ax.grid()
plt.title("Runtime vs Loops for %s input" % inp)
#plt.yticks(range(0,6000000,500000))
#plt.yticks(yticks)
#plt.xticks([0,4,8,12,16,20,24,28,32])

plt.legend()
out_png = 'graph_part2.png'
plt.savefig(out_png, dpi=150)
plt.show()
plt.close()
#plt.show()
    

#!/usr/bin/env python3
import os
import os.path
from subprocess import check_output
import re
from time import sleep
import filecmp

THREADS = [1,2,3,4,5,8,9,15,32,69,101,155,301]
LOOPS = [1,100,1000,10000,100000]
INPUTS = ["1k.txt","8k.txt", "16k.txt"]

csvs_threads = []
csvs_sequential = []
# generate sequential results files
for inp in INPUTS:
    for loop in LOOPS:
        output_file_name = "seq_%s_%d_loops.txt" %(inp,loop)
        if not os.path.isfile("results/%s" % output_file_name):
            cmd = "./bin/prefix_scan -o results/%s -n %d -i tests/%s -l %d" % (output_file_name,0,inp,loop)
            out = check_output(cmd, shell=True).decode("ascii")
# generate threaded results files 
for inp in INPUTS:
    for loop in LOOPS:
        for thr in THREADS:
            output_file_name = "thr_%d_%s_%d_loops.txt" % (thr,inp,loop)
            if not os.path.isfile("results/%s" % output_file_name):
                cmd = "./bin/prefix_scan -o results/%s -n %d -i tests/%s -l %d -s" % (output_file_name,thr,inp,loop)
                out = check_output(cmd, shell=True).decode("ascii")
# check results
for inp in INPUTS:
    for loop in LOOPS:
        seq_file_name = "results/seq_%s_%d_loops.txt" %(inp,loop)
        for thr in THREADS:
            thr_file_name = "results/thr_%d_%s_%d_loops.txt" % (thr,inp,loop)
            if os.path.isfile(seq_file_name) and os.path.isfile(thr_file_name):
                if not filecmp.cmp(seq_file_name,thr_file_name,shallow=False):
                    print("!!threaded out not equal to sequential out for -n %d -i %s -l %d!!"%(thr,inp,loop))
                else:
                    print("thr == seq for -n %d -i %s -l %d!"%(thr,inp,loop))
            else:
                print("a file does not exist for -n %d -i %s -l %d!"%(thr,inp,loop))
#ifndef _ARGPARSE_H
#define _ARGPARSE_H

#include <getopt.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>

const int sequential = 0;
const int cuda_basic = 1;
const int cuda_shmem = 2;
const int cuda_thrust = 3;

struct options_t {
    int num_cluster; // k
    int dims;
    char *in_file;
    int max_num_iter;
    double threshold;
    bool centroids;
    int seed;
    int version;
};

void get_opts(int argc, char **argv, struct options_t *opts);
#endif

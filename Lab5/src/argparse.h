#ifndef _ARGPARSE_H
#define _ARGPARSE_H

#include <getopt.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>

const int sequential = 0;
const int mpi = 1;

/*
-i inputfilename (char *): input file name
-o outputfilename (char *): output file name
-s steps (int): number of iterations
-t \theta(double): threshold for MAC
-d dt(double): timestep
-V: (OPTIONAL, see below) flag to turn on visualization window
-p: if true, run in parallel
*/

struct options_t {
    char *in_file;
    char *out_file;
    int steps;
    double theta;
    double dt;
    bool visualize;
    bool parallel;
};

void get_opts(int argc, char **argv, struct options_t *opts);
#endif

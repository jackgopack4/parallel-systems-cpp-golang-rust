#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>
#include <fstream>
#include <cstring>
#include <string>
#include <sstream>
#include <cstdio>
#include <getopt.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>

#include "kmeans_kernel.h"

int kmeans_rand();
void kmeans_srand(unsigned int seed);
void read_file(struct options_t* args,
               double*** points,
               int& num_points);
void print_output(bool clusters, double** p, double** c, int* labels, int num_points, int k, int dims);
void get_opts(int argc, char **argv, struct options_t *opts);
void assign_centers(double*** c, double** p, int k, int cmd_seed, int num_points, int dims);
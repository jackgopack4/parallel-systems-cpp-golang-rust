#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <sstream>
#include <cstdio>
#include <getopt.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>

int kmeans_rand();
void kmeans_srand(unsigned int seed);
void read_file(struct options_t* args,
               double*** points,
               int& num_points);
void print_output(bool clusters, double** p, double** c, int* labels, int num_points, int k, int dims);

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
void assign_centers(double*** c, double** p, int k, int cmd_seed, int num_points, int dims);
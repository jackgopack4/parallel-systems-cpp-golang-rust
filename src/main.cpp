#include <iostream>
#include <stdio.h>
#include "io.h"
#include <chrono>
#include <cstring>
#include "argparse.h"
//#include "operators.h"
#include "helpers.h"
#include "kmeans_cpu.h"
//#include "prefix_sum.h"
//#include "spin_barrier.h"

using namespace std;

int main(int argc, char **argv)
{
    // Parse args
    struct options_t opts;
    get_opts(argc, argv, &opts);
    struct points * input_vals;
    read_file(&opts,&input_vals); // also allocates input_vals
    
    struct centers * centroids = alloc_centers(opts.num_cluster, opts.dims);
    assign_centers(centroids,input_vals,opts.num_cluster,opts.seed);
    int* labels = compute_kmeans(&opts,input_vals,centroids);
    print_output(opts.centroids,input_vals,centroids,labels);
    free(labels);
    //free_centers(centroids);
    free_points(input_vals);
    
    return 0;
}

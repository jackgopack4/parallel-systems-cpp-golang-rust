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
    //struct points * input_vals;
    double** points;
    int k = opts.num_cluster;
    int dims = opts.dims;
    int cmd_seed = opts.seed;
    bool cluster_output = opts.centroids;
    /*
    if(!cluster_output) {
        printf("outputing assignment of every point on finish.\n");
    }
    else {
        printf("outputing final centroids on finish.\n");
    }
    */
    int num_points;
    read_file(&opts,&points,num_points); // also allocates input_vals
    
    //struct centers * centroids = alloc_centers(k, dims);
    double** centroids = (double**) malloc(k*sizeof(double*));
    for(auto i=0;i<k;++i) {
        centroids[i] = (double*) calloc(dims,sizeof(double));
    }
    int* indices = (int*) calloc(num_points,sizeof(int));
    
    assign_centers(&centroids,points,k,cmd_seed, num_points, dims);
    auto start = std::chrono::high_resolution_clock::now();
    compute_kmeans(&opts,points,&centroids,&indices,num_points);
    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    //std::cout << "time: " << diff.count() << std::endl;
     
    print_output(cluster_output,points,centroids,indices,num_points, k, dims);
    
    free(indices);
    //free_centers(centroids);
    for(auto i=0;i<k;++i) {
        free(centroids[i]);
    }
    free(centroids);
    //free_points(input_vals);
    for(auto i=0;i<num_points;++i) {
        free(points[i]);
    }
    free(points);
    return 0;
}

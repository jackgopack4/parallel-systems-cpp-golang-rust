#include <iostream>
#include <stdio.h>
#include "io.h"
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
    double* points;
    int k = opts.num_cluster;
    int dims = opts.dims;
    int cmd_seed = opts.seed;
    bool cluster_output = opts.centroids;
    //int v = opts.version;

    int num_points;
    read_file(&opts,&points,&num_points); // also allocates input_vals
    
    //struct centers * centroids = alloc_centers(k, dims);
    double* centroids = (double*) calloc(k*dims,sizeof(double));
    int* indices = (int*) calloc(num_points,sizeof(int));

    assign_centers(&centroids,points,k,cmd_seed, num_points, dims);    


    compute_kmeans(&opts,points,&centroids,&indices,num_points);
    print_output(cluster_output,points,centroids,indices,num_points, k, dims);

    
    //std::cout << "time: " << diff.count() << std::endl;
     
    
    free(indices);
    free(centroids);

    free(points);
    return 0;
}

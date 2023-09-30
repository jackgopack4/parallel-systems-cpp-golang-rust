#include <stdio.h>
#include "kmeans.h"

void helloFromGPU_wrapper(int blocks, int threads);
__global__ void helloFromGPU(void);
void compute_kmeans(options_t* opts, double** points, double*** centroids, int** labels, int num_points);
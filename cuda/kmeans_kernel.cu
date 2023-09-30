#include "kmeans_kernel.h"

void helloFromGPU_wrapper(int blocks, int threads) {
    helloFromGPU <<<blocks,threads>>>();
    cudaDeviceSynchronize();
}

__global__ void helloFromGPU(void) {
    printf("Hello World from GPU block %d, thread %d!\n",blockIdx.x,threadIdx.x);
}

void compute_kmeans(options_t* opts, double** points, double*** centroids, int** labels, int num_points) {
    printf("computing kmeans\n");
}
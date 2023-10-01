#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <chrono>

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

void helloFromGPU_wrapper(int blocks, int threads);
__global__ void helloFromGPU(void);
void compute_kmeans_cuda(options_t* opts, double** points, double*** centroids, int** labels, int num_points);
__global__ void findNearestCentroids_kernel(int* labels, double* points, double* centroids, double* old_centroids, int num_points, int k, int dims);
__device__ double euclideanDistance(double* point1, double* point2, int n);
__global__ void averageLabeledCentroids_kernel(double* points, int* labels, double* centroids, int* counts, int k, int dims, int num_points);
__global__ void updateCentroids_kernel(double* centroids, double* old_centroids, int* counts, int k, int dims, bool* centroid_changed, double tolerance);
__device__ bool hasConverged(double* old_centroids, double* new_centroids, int k, int dims, double tolerance);

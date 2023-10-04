#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <cmath>
#include <chrono>
#include <limits>
#include <float.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/extrema.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>
#include <thrust/for_each.h>
#include <thrust/tuple.h>
#include <thrust/inner_product.h>
#include <thrust/logical.h>

#define CUDA_CHECK_ERROR() checkCudaError(__FILE__, __LINE__)

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
void compute_kmeans_cuda(options_t* opts, double* points, double** centroids, int** labels, int num_points);
__global__ void findNearestCentroids_kernel(int* labels, double* points, double* centroids, double* old_centroids, double* distances, int num_points, int k, int dims, bool first_time);
__device__ double euclideanDistance(double* point1, double* point2, int k);
__global__ void averageLabeledCentroids_kernel(double* points, int* labels, double* centroids, double* old_centroids, int* counts, int k, int dims, int num_points);
__global__ void averageLabeledCentroids_shmem_kernel(double* points, int* labels, double* centroids, double* old_centroids,int* counts, int k, int dims, int num_points);
__global__ void updateCentroids_kernel(double* centroids, double* old_centroids, int* counts, int k, int dims, bool* centroid_changed, double tolerance);
__device__ bool hasConverged(double* old_centroids, double* new_centroids, int k, int dims, double tolerance);
__global__ void calcDistances_kernel(double* distances, double* points, double* centroids, int num_points, int k, int dims);
__global__ void calcDistances_shmem_kernel(double* distances, double* points, double* centroids, int num_points, int k, int dims);
void findMinIndices(const thrust::device_vector<double>& distances, thrust::device_vector<int>& min_indices, int n, int k);
inline void checkCudaError(const char *file, int line);
bool checkDistanceThreshold(thrust::device_vector<double>& thrust_centroids, thrust::device_vector<double>& thrust_old_centroids, double threshold);
// Euclidean distance functor
struct EuclideanDistanceFunctor : public thrust::binary_function<double, double, double>
{
    __host__ __device__
    double operator()(const double& a, const double& b) const
    {
        double diff = a - b;
        return diff * diff;
    }
};

// Check if distance exceeds threshold functor
struct CheckThresholdFunctor
{
    double threshold;

    CheckThresholdFunctor(double threshold) : threshold(threshold) {}

    __host__ __device__
    bool operator()(const double& distance) const
    {
        return distance > threshold;
    }
};
// calculate distances between two points
struct CalculateDistancesFunctor
{
    const double* points_n;
    const double* points_k;
    int k, dims;

    CalculateDistancesFunctor(const thrust::device_vector<double>& points_n, const thrust::device_vector<double>& points_k, int k, int dims)
        : points_n(thrust::raw_pointer_cast(points_n.data())), points_k(thrust::raw_pointer_cast(points_k.data())), k(k), dims(dims) {}

    __host__ __device__
    double operator()(const int& idx) const
    {
        int point_n_idx = idx / k;
        int point_k_idx = idx % k;

        const double* point_n = points_n + point_n_idx * dims;
        const double* point_k = points_k + point_k_idx * dims;

        // Calculate the Euclidean distance between point_n and point_k
        double distance = 0.0;
        for (int i = 0; i < dims; ++i) {
            double diff = point_n[i] - point_k[i];
            distance += diff * diff;
        }

        return sqrt(distance);
    }
};

// Functor to calculate the relative index of the minimum value in a segment
struct MinIndexFunctor : public thrust::unary_function<thrust::tuple<double, int>, int>
{
    __host__ __device__
    int operator()(const thrust::tuple<double, int>& tuple) const
    {
        return thrust::get<1>(tuple);
    }
};

// Functor to update centroids based on points and labels
struct UpdateCentroidsFunctor
{
    int k;
    int dims;
    double* centroids_ptr; // raw pointer to centroids data

    UpdateCentroidsFunctor(int k, int dims, double* centroids_ptr) : k(k), dims(dims), centroids_ptr(centroids_ptr) {}

    __host__ __device__
    void operator()(const thrust::tuple<double, int>& point_label_tuple)
    {
        int label = thrust::get<1>(point_label_tuple);
        double point_value = thrust::get<0>(point_label_tuple);

        // Update the corresponding centroid value
        centroids_ptr[label * dims] += point_value;
    }
};

// Functor for element-wise division
struct DivideFunctor : public thrust::binary_function<double, int, double>
{
    __host__ __device__
    double operator()(const double& x, const int& y) const
    {
        return x / static_cast<double>(y);
    }
};

struct CountLabelsFunctor
{
    int k;

    CountLabelsFunctor(int k) : k(k) {}

    __host__ __device__
    int operator()(int label)
    {
        return 1;
    }
};
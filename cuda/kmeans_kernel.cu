#include "kmeans_kernel.h"

void helloFromGPU_wrapper(int blocks, int threads) {
    helloFromGPU <<<blocks,threads>>>();
    cudaDeviceSynchronize();
}

__global__ void helloFromGPU(void) {
    printf("Hello World from GPU block %d, thread %d!\n",blockIdx.x,threadIdx.x);
}

void compute_kmeans_cuda(options_t* opts, double* points, double** centroids, int** labels, int num_points) {
    /*
    printf("computing kmeans\n");
    printf("sizeof double: %lu\n",sizeof(double));
    for(auto i=0;i<num_points;++i) {
        printf("sizeof point %d: %lu\n",i,sizeof(points[i]));
        printf("0x%p\n", &points[i]);
    }
    helloFromGPU <<<1,10>>>();
    cudaDeviceSynchronize();
    */
      /* book-keeping */
    int iterations = 0;
    int k = opts->num_cluster;
    int dims = opts->dims;
    int max_num_iter = opts->max_num_iter;
    /* core algorithm */
    bool done = false;
    double tolerance = opts->threshold;
    /* Allocate device memory */
    double *d_points, *d_centroids, *d_old_centroids;
    int *d_labels, *d_counts;

    cudaMalloc((void**)&d_points, num_points * dims * sizeof(double));
    cudaMalloc((void**)&d_centroids, k * dims * sizeof(double));
    cudaMalloc((void**)&d_old_centroids, k * dims * sizeof(double));
    cudaMalloc((void**)&d_labels, num_points * sizeof(int));
    cudaMalloc((void**)&d_counts, k * sizeof(int));

    // allocate unified memory
    bool *centroid_changed;
    cudaMallocManaged(&centroid_changed, k*sizeof(bool));

    /* Copy data from host to device */
    cudaMemcpy(d_points, points, num_points * dims * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, *centroids, k * dims * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_old_centroids, *centroids, k * dims * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, (*labels), num_points * sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    while (!done) {
        auto start = std::chrono::high_resolution_clock::now();

        ++iterations;

        /* Launch CUDA kernels */
        findNearestCentroids_kernel<<<(num_points + 255) / 256, 256>>>(d_labels, d_points, d_centroids, d_old_centroids, num_points, k, dims);
        cudaDeviceSynchronize();  // Wait for the kernel to finish

        averageLabeledCentroids_kernel<<<(num_points + 255) / 256, 256>>>(d_points, d_labels, d_centroids, d_counts, k, dims, num_points);
        cudaDeviceSynchronize();  // Wait for the kernel to finish

        updateCentroids_kernel<<<(k + 255) / 256, 256>>>(d_centroids, d_old_centroids, d_counts, k, dims, centroid_changed, tolerance);
        cudaDeviceSynchronize();  // Wait for the kernel to finish
        done = true;
        for(int i=0; i<k; ++i) {
            if (!centroid_changed[k]) {
                done = false;
            }
        }
        if(iterations == max_num_iter) {
            done = true;
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        printf("runtime for iteration %d: %ld\n",iterations,diff.count());
    }
    printf("total iterations: %d\n",iterations);

    // Copy data back to host
    cudaMemcpy(*centroids, d_centroids, k * dims * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(*labels, d_labels, num_points * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_old_centroids);
    cudaFree(d_labels);
    cudaFree(d_counts);
    cudaFree(centroid_changed);
}

__device__ double euclideanDistance(double* point1, double* point2, int n) {
    double sum = 0.0;

    for (int i = 0; i < n; i++) {
        double diff = (point1[i]) - (point2[i]);
        sum += diff * diff;
    }

    return sqrt(sum);
}


__global__ void findNearestCentroids_kernel(int* labels, double* points, double* centroids, double* old_centroids, int num_points, int k, int dims) {
    int idx = blockIdx.x *blockDim.x +threadIdx.x;

    if (idx < num_points) {
        int label = labels[idx];
        double distance = euclideanDistance(&points[idx * dims], &centroids[label * dims], dims);

        for (int j = 0; j < k; ++j) {
            double tmp = euclideanDistance(&points[idx * dims], &centroids[j * dims], dims);
            if (tmp < distance) {
                distance = tmp;
                label = j;
            }
        }
        labels[idx] = label;
        memcpy((void**)&old_centroids[idx * dims], (void**)&centroids[idx * dims], dims* sizeof(double));
        memset((void**)&centroids[idx * dims], 0, k * sizeof(double));	
    }
}

__global__ void averageLabeledCentroids_kernel(double* points, int* labels, double* centroids, int* counts, int k, int dims, int num_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_points) {
        atomicAdd(&counts[labels[idx]], 1);
        for (int j = 0; j < dims; ++j) {
            atomicAdd(&centroids[labels[idx] * dims + j], points[idx * dims + j]);
        }
    }
}
/* have to break out this step due to kernel synchronization */
__global__ void updateCentroids_kernel(double* centroids, double* old_centroids, int* counts, int k, int dims, bool* centroid_changed, double tolerance) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < k) {
        if (counts[idx] > 0) {
            for (int j = 0; j < dims; ++j) {
                centroids[idx * dims + j] /= counts[idx];
            }
        }
        centroid_changed[idx] = hasConverged(old_centroids,centroids,k,dims, tolerance);
    }
}

__device__ bool hasConverged(double* old_centroids, double* new_centroids, int k, int dims, double tolerance) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < k) {
        for (int j=0; j < dims; ++j) {
            double diff = fabs((old_centroids)[idx*dims + j] - (new_centroids)[idx * dims + j]);
            if (diff > tolerance) {
                return false;
            }
        }
        return true;
    }
    return true;
}

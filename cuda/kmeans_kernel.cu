#include "kmeans_kernel.h"

void helloFromGPU_wrapper(int blocks, int threads) {
    helloFromGPU <<<blocks,threads>>>();
    cudaDeviceSynchronize();
}

__global__ void helloFromGPU(void) {
    printf("Hello World from GPU block %d, thread %d!\n",blockIdx.x,threadIdx.x);
}

void compute_kmeans_cuda(options_t* opts, double* points, double** centroids, int** labels, int num_points) {
      /* book-keeping */
    int iterations = 0;
    int k = opts->num_cluster;
    int dims = opts->dims;
    int max_num_iter = opts->max_num_iter;
    /* core algorithm */
    bool done = false;
    double tolerance = opts->threshold;
    /* Allocate device memory */
    double *d_points, *d_centroids, *d_old_centroids, *d_distances;
    int *h_counts = (int*) calloc(k,sizeof(int));
    int *d_labels, *d_counts;

    cudaMalloc((void**)&d_points, num_points * dims * sizeof(double));
    cudaMalloc((void**)&d_centroids, k * dims * sizeof(double));
    cudaMalloc((void**)&d_old_centroids, k * dims * sizeof(double));
    cudaMalloc((void**)&d_distances, num_points * k * sizeof(double));
    cudaMalloc((void**)&d_labels, num_points * sizeof(int));
    cudaMalloc((void**)&d_counts, k * sizeof(int));

    // allocate unified memory
    bool *centroid_changed;
    cudaMallocManaged(&centroid_changed, k*sizeof(bool));
    for(int i=0;i<k;++i) {
        centroid_changed[i] = true;
    }
    /* Copy data from host to device */
    cudaMemcpy(d_points, points, num_points * dims * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, *centroids, k * dims * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_old_centroids, *centroids, k * dims * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, (*labels), num_points * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_counts,h_counts,k*sizeof(int),cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    
    while (!done) {
        auto start = std::chrono::high_resolution_clock::now();

        ++iterations;

        /* Launch CUDA kernels */
        calcDistances_kernel<<<(num_points*k+255) / 256, 256>>>(&d_distances,d_points, d_centroids, num_points, k, dims);
        cudaDeviceSynchronize();

        findNearestCentroids_kernel<<<(num_points + 255) / 256, 256>>>(&d_labels, d_points, d_centroids, d_old_centroids, d_distances, num_points, k, dims);
        cudaDeviceSynchronize();  // Wait for the kernel to finish

        averageLabeledCentroids_kernel<<<(num_points + 255) / 256, 256>>>(d_points, d_labels, &d_centroids, &d_counts, k, dims, num_points);
        cudaDeviceSynchronize();  // Wait for the kernel to finish

        updateCentroids_kernel<<<(k + 7)/8, 8>>>(&d_centroids, d_old_centroids, d_counts, k, dims, &centroid_changed, tolerance);
        cudaDeviceSynchronize();  // Wait for the kernel to finish
        done = true;
        for(int i=0; i<k; ++i) {
            if (centroid_changed[i]) {
                /*printf("centroid %d changed\n",i);*/
                done = false;
            }
        }
        if(iterations == max_num_iter) {
            done = true;
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        /*printf("runtime for iteration %d: %ld\n",iterations,diff.count());*/
    }
    /*printf("total iterations: %d\n",iterations);*/

    // Copy data back to host
    cudaMemcpy(*centroids, d_centroids, k * dims * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(*labels, d_labels, num_points * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Free device memory
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_old_centroids);
    cudaFree(d_distances);
    cudaFree(d_labels);
    cudaFree(d_counts);
    cudaFree(centroid_changed);
    free(h_counts);
}

__device__ double euclideanDistance(double* point1, double* point2, int k) {
    double sum = 0.0;

    for (int i = 0; i < k; i++) {
        double diff = (point1[i]) - (point2[i]);
        sum += diff * diff;
    }

    return sqrt(sum);
}

__global__ void calcDistances_kernel(double** distances, double* points, double* centroids, int num_points, int k, int dims) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < num_points * k) {
        int centroid_idx = idx % num_points;
        int point_idx = (idx - centroid_idx) / num_points;
        (*distances)[idx] = euclideanDistance(&points[point_idx * dims], &centroids[centroid_idx * dims], dims);
    }
}

__global__ void findNearestCentroids_kernel(int** labels, double* points, double* centroids, double* old_centroids, double* distances, int num_points, int k, int dims) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_points) {
        int label = (*labels)[idx];
        /*double distance = euclideanDistance(&points[idx * dims], &centroids[label * dims], dims); */
        int distances_idx = idx * num_points + label;
        double distance = distances[distances_idx];
        for (int j = 0; j < k; ++j) {
            if (j == label) {
                continue;
            }
            double tmp = distances[idx*num_points + j];
            if (tmp < distance) {
                distance = tmp;
                label = j;
            }
        }
        (*labels)[idx] = label;
        memcpy((void**)&old_centroids[idx * dims], (void**)&centroids[idx * dims], dims* sizeof(double));
        memset((void**)&centroids[idx * dims], 0, dims * sizeof(double));	
    }
}

__global__ void averageLabeledCentroids_kernel(double* points, int* labels, double** centroids, int** counts, int k, int dims, int num_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_points) {
        atomicAdd(&(*counts)[labels[idx]], 1);
        for (int j = 0; j < dims; ++j) {
            atomicAdd(&(*centroids)[labels[idx] * dims + j], points[idx * dims + j]);
        }
    }
}
/* have to break out this step due to kernel synchronization */
__global__ void updateCentroids_kernel(double** centroids, double* old_centroids, int* counts, int k, int dims, bool** centroid_changed, double tolerance) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    /*printf("updateCentroids called for idx %d\n",idx);*/
    if (idx < k) {
        if (counts[idx] > 0) {
            for (int j = 0; j < dims; ++j) {
                (*centroids)[idx * dims + j] /= counts[idx];
            }
        }
        (*centroid_changed)[idx] = !hasConverged(old_centroids,*centroids,k,dims, tolerance);
    }
}

__device__ bool hasConverged(double* old_centroids, double* new_centroids, int k, int dims, double tolerance) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    /*printf("hasConverged called for idx %d\n",idx);*/
    if (idx < k) {
        for (int j=0; j < dims; ++j) {
            double diff = fabs((old_centroids)[idx*dims + j] - (new_centroids)[idx * dims + j]);
            if (diff > tolerance) {
                /*printf("diff %f more than tolerance %f, centroid %d\n",diff,tolerance,idx);*/
                return false;
            }
        }
        /*printf("diff less than tolerance %f, centroid %d\n",tolerance,idx);*/
        return true;
    }
    return true;
}

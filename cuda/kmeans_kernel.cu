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
    double *h_distances = (double*) calloc(num_points*k,sizeof(double));
    //printf("size of distances array: %d\n",num_points*k);
    for(int i=0;i<num_points*k;++i) {
        //printf("distances[%d]: %f\n",i,h_distances[i]);
    }
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
    cudaMemcpy(d_distances,h_distances,num_points*k*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, (*labels), num_points * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_counts,h_counts,k*sizeof(int),cudaMemcpyHostToDevice);
    
    cudaDeviceSynchronize();
    
    auto start = std::chrono::high_resolution_clock::now();
    while (!done) {

        ++iterations;
        bool first_time = (iterations == 1);
        /* Launch CUDA kernels */
        calcDistances_kernel<<<(num_points*k+255) / 256, 256>>>(d_distances,d_points, d_centroids, num_points, k, dims);
        cudaDeviceSynchronize();
        CUDA_CHECK_ERROR();

        findNearestCentroids_kernel<<<(num_points + 127) / 128, 128>>>(d_labels, d_points, d_centroids, d_old_centroids, d_distances, num_points, k, dims, first_time);
        cudaDeviceSynchronize();  // Wait for the kernel to finish
        CUDA_CHECK_ERROR();

        cudaMemcpy(d_old_centroids,d_centroids,k*dims*sizeof(double),cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
        CUDA_CHECK_ERROR();
        cudaMemset(d_centroids,0,k*dims*sizeof(double));
        cudaMemset(d_counts,0,k*sizeof(int));
        cudaDeviceSynchronize();
        CUDA_CHECK_ERROR();

        averageLabeledCentroids_kernel<<<(num_points + 127) / 128, 128>>>(d_points, d_labels, d_centroids, d_old_centroids, d_counts, k, dims, num_points);
        cudaDeviceSynchronize();  // Wait for the kernel to finish
        CUDA_CHECK_ERROR();
        //printf("about to call updateCentroids_kernel\n");
        updateCentroids_kernel<<<(k + 7)/8, 8>>>(d_centroids, d_old_centroids, d_counts, k, dims, centroid_changed, tolerance);
        cudaDeviceSynchronize();  // Wait for the kernel to finish
        CUDA_CHECK_ERROR();
        //printf("finished updateCentroids_kernel\n");
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
        /*printf("runtime for iteration %d: %ld\n",iterations,diff.count());*/
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double,std::ratio<1,1000>> diff = end - start;
    printf("%d,%lf\n", iterations, diff.count()/iterations);
        
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

__global__ void calcDistances_kernel(double* distances, double* points, double* centroids, int num_points, int k, int dims) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("calcDistances_kernel called for idx %d\n",idx);
    if(idx < num_points * k) {
        int centroid_idx = idx % k;
        int point_idx = (idx - centroid_idx) / k;

        distances[idx] = euclideanDistance(&points[point_idx * dims], &centroids[centroid_idx * dims], dims);
        //printf("distance set for point %d centroid %d as %f\n",point_idx,centroid_idx,distances[idx]);
    }
}

__global__ void findNearestCentroids_kernel(int* labels, double* points, double* centroids, double* old_centroids, double* distances, int num_points, int k, int dims, bool first_time) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("called findNearestCentroids_kernel for idx %d\n",idx);
    if (idx < num_points) {
        int label = labels[idx];
        //printf("label = %d\n",label);
        double distance;
        if(first_time) {
            distance = DBL_MAX;
        }
        else {
            int distances_idx = idx * k + label;
            distance = distances[distances_idx];
            //printf("initial distance for point %d labeled %d: %f\n",idx,label,distance);
        }
        /*double distance = euclideanDistance(&points[idx * dims], &centroids[label * dims], dims); */
        for (int j = 0; j < k; ++j) {
            if (!first_time && j == label) {
                continue;
            }
            double tmp = distances[idx*k + j];
            //printf("dist from point %d [%f, %f] to centroid %d [%f, %f] = %f\n",idx,points[idx*dims],points[idx*dims+1],j,centroids[j*dims],centroids[j*dims+1],tmp);
            if (tmp < distance) {
                distance = tmp;
                label = j;
            }
        }
        //printf("closer distance for point [%f,%f] at centroid %d: [%f, %f]\n",points[idx*dims],points[idx*dims+1],label,centroids[label*dims],centroids[label*dims+1]);
        labels[idx] = label;
        //printf("set label for point %d as centroid %d\n",idx,label);
        //memcpy((void**)&old_centroids[idx * dims], (void**)&centroids[idx * dims], dims* sizeof(double));
        //memset((void**)&centroids[idx * dims], 0, dims * sizeof(double));	
    }
}

__device__ double euclideanDistance(double* point1, double* point2, int k) {
    double sum = 0.0;

    for (int i = 0; i < k; i++) {
        double diff = (point1[i]) - (point2[i]);
        sum += diff * diff;
    }

    return sqrt(sum);
}




__global__ void averageLabeledCentroids_kernel(double* points, int* labels, double* centroids, double* old_centroids,int* counts, int k, int dims, int num_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("averageLabeledCentroids_kernel called for idx %d\n",idx);
    if (idx < num_points) {
        //printf("label for point %d is centroid %d\n",idx,labels[idx]);
        atomicAdd(&counts[labels[idx]], 1);
        //printf("counts at %d after add is %d\n",labels[idx],counts[labels[idx]]);
        for (int j = 0; j < dims; ++j) {
            //printf("dim %d value for point %d = %f, for centroid %d = %f\n",j,idx,points[idx*dims+j],labels[idx],old_centroids[labels[idx] * dims + j]);
            atomicAdd(&centroids[labels[idx] * dims + j], points[idx * dims + j]);
        }
    }
}
/* have to break out this step due to kernel synchronization */
__global__ void updateCentroids_kernel(double* centroids, double* old_centroids, int* counts, int k, int dims, bool* centroid_changed, double tolerance) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("updateCentroids called for idx %d\n",idx);
    if (idx < k) {
        if (counts[idx] > 0) {
            for (int j = 0; j < dims; ++j) {
                centroids[idx * dims + j] /= counts[idx];
            }
        }
        centroid_changed[idx] = !hasConverged(old_centroids,centroids,k,dims, tolerance);
    }
}

__device__ bool hasConverged(double* old_centroids, double* new_centroids, int k, int dims, double tolerance) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    /*printf("hasConverged called for idx %d\n",idx);*/
    if (idx < k) {
        double diff = euclideanDistance(&old_centroids[idx*dims],&new_centroids[idx*dims],dims);
        return (diff <= tolerance);
    }
    return true;
}


inline void checkCudaError(const char *file, int line) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << file << ":" << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

#include "kmeans_cpu.h"

using namespace std;

void compute_kmeans(options_t* opts, double* points, double** centroids, int** labels, int num_points) {
    // book-keeping
    int iterations = 0;
    int k = opts->num_cluster;
    int dims = opts->dims;
    int max_num_iter = opts->max_num_iter;
    // core algorithm
    bool done = false;
    double* old_centroids = (double*) calloc(k*dims,sizeof(double));
    double tolerance = opts->threshold;
    auto start = std::chrono::high_resolution_clock::now();
    while(!done) {

        ++iterations;

        // labels is a mapping from each point in the dataset 
        // to the nearest (euclidean distance) centroid
        bool first_time = (iterations == 1);
        findNearestCentroids(labels, points, *centroids,num_points,k,dims,first_time);

        // the new centroids are the average of 
        // all the points that map to each centroid
        memcpy(old_centroids, (*centroids), k*dims*sizeof(double));
        memset((*centroids),0,k*dims*sizeof(double));

        averageLabeledCentroids(points, *labels, centroids,k,dims,num_points);
        if (iterations == max_num_iter || hasConverged(old_centroids, *centroids, k, dims, tolerance)) {
            done = true; // Convergence achieved, exit the loop
        }
        /*
        printf("old centroids, iteration %d:\n",iterations);
        for (int i=0;i<k;++i) {
            printf("%d",i);
            for (int j=0;j<dims; ++j) {
                printf(" %f",old_centroids[i*dims + j]);
            }
            printf("\n");
        }

        printf("new centroids, iteration %d:\n",iterations);
        for (int i=0;i<k;++i) {
            printf("%d",i);
            for (int j=0;j<dims; ++j) {
                printf(" %f",(*centroids)[i*dims + j]);
            }
            printf("\n");
        }
        */

    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double,std::ratio<1,1000>> diff = end - start;
    printf("%d,%lf\n", iterations, diff.count()/iterations);
    free(old_centroids);
  
}

void findNearestCentroids(int** labels, double* p, double* c, int num_points, int k, int dims, bool first_time) {
    for(auto i=0;i<num_points;++i) {
        int label = (*labels)[i];
        double distance;
        if(first_time) {
            distance = DBL_MAX;
        }
        else {
            distance = euclideanDistance(&p[i*dims],&c[label*dims],dims);
        }
        for(auto j=0;j<k;++j) {
            if (!first_time && j == label) {
                continue;
            }
            double tmp = euclideanDistance(&p[i*dims],&c[j*dims],dims);
            if(tmp < distance) {
                distance = tmp;
                label = j;
            }
        }
        //printf("nearest centroid for point %d: idx %d\n",i,label);
        (*labels)[i] = label;
    }
}

bool hasConverged(double* old_centroids, double* new_centroids, int k, int dims, double tolerance) {
    for (int i = 0; i < k; ++i) {
        double diff = euclideanDistance(&old_centroids[i*dims],&new_centroids[i*dims],dims);
        if (diff > tolerance) {
            return false;
        }
    }
    return true; // Converged
}
void averageLabeledCentroids(double* p, int* labels, double** c, int k, int dims, int num_points) {
    int* counts = (int*) calloc(k,sizeof(int));

    for(auto i=0;i<num_points;++i) {
        counts[labels[i]] += 1;
        for(auto j=0;j<dims;++j) {
            (*c)[labels[i] * dims + j] += p[i * dims + j];
        }
    }
    for(auto i=0;i<k;++i) {
        if (counts[i] > 0) {
            for(auto j=0;j<dims;++j) {
                (*c)[i*dims + j] /= counts[i];
            }
        }
    }

    free(counts);
}

double euclideanDistance(double* point1, double* point2, int n) {
    double sum = 0.0;

    for (int i = 0; i < n; i++) {
        double diff = (point1[i]) - (point2[i]);
        sum += diff * diff;
    }

    return sqrt(sum);
}


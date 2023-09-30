#include "kmeans_cpu.h"

using namespace std;

void compute_kmeans(options_t* opts, double** points, double*** centroids, int** labels, int num_points) {
    // book-keeping
  int iterations = 0;
  int k = opts->num_cluster;
  int dims = opts->dims;\
  // core algorithm
  bool done = false;
  double** old_centroids = (double**) malloc(k*sizeof(double*));
  double tolerance = 0.001;
  for(auto i=0;i<k;++i) {
    old_centroids[i] = (double*) calloc(dims,sizeof(double));
  }
  while(!done) {\
    ++iterations;

    // labels is a mapping from each point in the dataset 
    // to the nearest (euclidean distance) centroid
    findNearestCentroids(labels, points, *centroids,num_points,k,dims);

    // the new centroids are the average 
    // of all the points that map to each 
    // centroid
    for (auto i = 0; i < k; ++i) {
        memcpy(old_centroids[i], (*centroids)[i], dims * sizeof(double));
    }
    averageLabeledCentroids(points, *labels, centroids,k,dims,num_points);
    if (hasConverged(&old_centroids, centroids, k, dims, tolerance)) {
        done = true; // Convergence achieved, exit the loop
    }
  }
  //std::cout << "iterations: " << iterations << std::endl;
  for(auto i=0;i<k;++i) {
    free(old_centroids[i]);// = (double*) calloc(dims,sizeof(double));
  }
  free(old_centroids);
  
}
bool hasConverged(double*** old_centroids, double*** new_centroids, int k, int dims, double tolerance) {
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < dims; ++j) {
            double diff = fabs((*old_centroids)[i][j] - (*new_centroids)[i][j]);
            if (diff > tolerance) {
                return false; // If any centroid has changed more than tolerance, not converged
            }
        }
    }
    return true; // Converged
}
void averageLabeledCentroids(double** p, int* labels, double*** c, int k, int dims, int num_points) {
    int* counts = (int*) calloc(k,sizeof(int));
    double** new_centroids = (double**) malloc(k*sizeof(double*));
    for(auto i=0;i<k;++i) {
        new_centroids[i] = (double*) calloc(dims,sizeof(double));
    }
    //new_centroids->num_centers = k;
    for(auto i=0;i<num_points;++i) {
        counts[labels[i]] += 1;
        for(auto j=0;j<dims;++j) {
            new_centroids[labels[i]][j] += p[i][j];
        }
    }
    for(auto i=0;i<k;++i) {
        if (counts[i] > 0) {
            for(auto j=0;j<dims;++j) {
                new_centroids[i][j] /= counts[i];
            }
        }
    }
    for (auto i = 0; i < k; ++i) {
        memcpy((*c)[i], new_centroids[i], dims * sizeof(double));
    }
    for(auto i=0;i<k;++i) {
        free(new_centroids[i]);
    }
    free(new_centroids);


    free(counts);
}

// Function to calculate the Euclidean distance between two n-dimensional points
double euclideanDistance(double* point1, double* point2, int n) {
    double sum = 0.0;

    for (int i = 0; i < n; i++) {
        double diff = (point1[i]) - (point2[i]);
        sum += diff * diff;
    }

    return sqrt(sum);
}

void findNearestCentroids(int** labels, double** p, double** c, int num_points, int k, int dims) {
    for(auto i=0;i<num_points;++i) {
        int label = (*labels)[i];
        double distance = euclideanDistance(p[i],c[label],dims);
        for(auto j=0;j<k;++j) {
            double tmp = euclideanDistance(p[i],c[j],dims);
            if(tmp < distance) {
                distance = tmp;
                label = j;
            }
        }
        (*labels)[i] = label;
    }
}
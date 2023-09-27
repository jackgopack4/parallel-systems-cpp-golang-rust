#include "kmeans_cpu.h"

using namespace std;

void compute_kmeans(options_t* opts, points* input_vals, centers* centroids) {
    // book-keeping
  int iterations = 0;
  centers* oldCentroids = alloc_centers(opts->num_cluster,opts->dims);
  int* labels = (int*) malloc((input_vals->num_points)*sizeof(int));
  // core algorithm
  while(iterations < 100) {

    memcpy(oldCentroids->centers, centroids->centers, (opts->num_cluster)*sizeof(double*));
    oldCentroids->num_centers = centroids->num_centers;
    ++iterations;

    // labels is a mapping from each point in the dataset 
    // to the nearest (euclidean distance) centroid
    findNearestCentroids(labels, input_vals, centroids);

    // the new centroids are the average 
    // of all the points that map to each 
    // centroid
    /*
    centroids = averageLabeledCentroids(dataSet, labels, k);
    done = iterations > MAX_ITERS || converged(centroids, oldCentroids);
    */
  }
  free(labels);
  free_centers(oldCentroids);
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

void findNearestCentroids(int* labels, points* p, centers* c) {
    for(auto i=0;i<p->num_points;++i) {
        int label = 0;
        double sample_distance = std::numeric_limits<double>::max();
        for(auto j=0;j<c->num_centers;++j) {
            double * point_coords =  p->points_array[i].coords_array;
            double tmp = euclideanDistance(point_coords,c->centers[j],p->dims);
            if(tmp < sample_distance) {
                sample_distance = tmp;
                label = j;
            }
        }
        labels[i] = label;
    }
}
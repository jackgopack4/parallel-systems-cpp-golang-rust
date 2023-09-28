#include "kmeans_cpu.h"

using namespace std;

int* compute_kmeans(options_t* opts, points* input_vals, centers* centroids) {
    // book-keeping
  int iterations = 0;
  //int k = centroids->num_centers;
  //int dims = input_vals->dims;
  int num_points = input_vals->num_points;
  //centers* oldCentroids = alloc_centers(k,dims);
  int* labels = (int*) malloc((num_points)*sizeof(int));
  // core algorithm
  while(iterations < 3) {
    /*
    for(auto i=0;i<k;++i) {
        memcpy(oldCentroids->centers[i],centroids->centers[i],(dims)*sizeof(double));
    }
    
    //memcpy(oldCentroids->centers, centroids->centers, (k)*sizeof(double*));
    oldCentroids->num_centers = centroids->num_centers;
    */
    ++iterations;

    // labels is a mapping from each point in the dataset 
    // to the nearest (euclidean distance) centroid
    findNearestCentroids(labels, input_vals, centroids);

    // the new centroids are the average 
    // of all the points that map to each 
    // centroid
    
    centroids = averageLabeledCentroids(input_vals, labels, centroids);
    /*
    done = iterations > MAX_ITERS || converged(centroids, oldCentroids);
    */
  }
  //free(labels);
  //free_centers(oldCentroids);
  return labels;
}

centers* averageLabeledCentroids(points* p, int* labels, centers* c) {
    int k = c->num_centers;
    int dims = p->dims;
    int num_points = p->num_points;
    int* counts = (int*) calloc(k,sizeof(int));
    centers* new_centroids = alloc_centers(k,dims);
    new_centroids->num_centers = k;
    for(auto i=0;i<num_points;++i) {
        int old_centers_index = labels[i];
        for(auto j=0;j<dims;++j) {
            new_centroids->centers[old_centers_index][j] += c->centers[old_centers_index][j];
            counts[old_centers_index] += 1;
        }
    }
    for(auto i=0;i<k;++i) {
        if (counts[i] > 0) {
            for(auto j=0;j<dims;++j) {
                new_centroids->centers[i][j] /= counts[i];
            }
        }
    }


    free_centers(c);
    free(counts);
    return new_centroids;
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
    int num_points = p->num_points;
    int k = c->num_centers;
    for(auto i=0;i<num_points;++i) {
        int label = 0;
        double sample_distance = std::numeric_limits<double>::max();
        for(auto j=0;j<k;++j) {
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
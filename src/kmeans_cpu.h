#include <iostream>
#include <math.h>
#include <float.h>
#include <limits>
#include <string.h>
#include <chrono>

#include "argparse.h"

void compute_kmeans(options_t* opts, double** points, double*** centroids, int** labels, int num_points);
double euclideanDistance(double* point1, double* point2, int n);
void findNearestCentroids(int** labels, double** p, double** c, int num_points, int k, int dims);
void averageLabeledCentroids(double** p, int* labels, double*** c, int k, int dims, int num_points);
bool hasConverged(double*** old_centroids, double*** new_centroids, int k, int dims, double tolerance);
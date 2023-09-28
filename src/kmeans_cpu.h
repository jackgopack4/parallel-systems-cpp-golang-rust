#include <iostream>
#include <math.h>
#include <float.h>
#include <limits>
#include <string.h>

#include "helpers.h"
#include "argparse.h"

int* compute_kmeans(options_t* opts, points* input_vals, centers* centroids);
double euclideanDistance(double* point1, double* point2, int n);
void findNearestCentroids(int* labels, points* p, centers* c);
centers* averageLabeledCentroids(points* p, int* labels, centers* c);
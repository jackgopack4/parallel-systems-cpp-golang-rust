#pragma once

#include <stdlib.h>
#include <string.h>

struct centers {
    int num_centers;
    double** centers;
};

struct point {
    int centroid_index;
    double* coords_array;
};

struct points {
    int dims;
    int num_points;
    point* points_array;
};
void free_centers(centers* c);
centers* alloc_centers(int k, int dims);
void assign_centers(centers* c, points* p, int k, int cmd_seed);
struct points* alloc_points(int dims, int num_points);
void free_points(points* p);
int kmeans_rand();
void kmeans_srand(unsigned int seed);
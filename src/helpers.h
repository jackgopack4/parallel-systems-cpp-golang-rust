#pragma once

#include <stdlib.h>

struct point {
    double* coords_array;
};

struct points {
    int dims;
    int num_points;
    point* points_array;
};

struct points* alloc_points(int dims, int num_points);

void free_points(points* p);
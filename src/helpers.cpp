#include "helpers.h"

points* alloc_points(int dims, int num_points) {
    points* p = (points*) malloc(sizeof(points));
    p->dims = dims;
    p->num_points = num_points;
    p->points_array = (point*) malloc((num_points)*sizeof(point));
    for (int i = 0; i<num_points; ++i) {
        p->points_array[i].coords_array = (double*) calloc(dims,sizeof(double));
    }
    return p;
}

void free_points(points* p) {
    int num_points = p->num_points;
    for (int i=0; i<num_points;++i) {
        free(p->points_array[i].coords_array);
    }
    free(p->points_array);
    free(p);
}


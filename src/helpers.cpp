#include "helpers.h"

centers* alloc_centers(int k, int dims) {
    centers* c = (centers*) malloc(sizeof(centers));
    c->centers = (double**) malloc((k)*sizeof(double*));
    for(auto i=0;i<k;++i) {
        c->centers[i] = (double*) calloc(dims,sizeof(double));
    } 
    return c;
}

void free_centers(centers* c) {
    for(auto i=0; i<c->num_centers; ++i) {
        free(c->centers[i]);
    } // don't need to free because it's just a pointer to the points struct
    free(c->centers);
    free(c);
}

void assign_centers(centers* c, points* p, int k, int cmd_seed) {
    c->num_centers = k;
    kmeans_srand(cmd_seed); // cmd_seed is a cmdline arg
    for (int i=0; i<c->num_centers; i++){
        int index = kmeans_rand() % p->num_points;
        // you should use the proper implementation of the following
        // code according to your data structure
        memcpy(c->centers[i],p->points_array[index].coords_array,(p->dims)*sizeof(double));
        //*c->centers[i] = p->points_array[index].coords_array;
    }
}

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
static unsigned long int next = 1;
static unsigned long kmeans_rmax = 32767;

int kmeans_rand() {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next/65536) % (kmeans_rmax+1);
}
void kmeans_srand(unsigned int seed) {
    next = seed;
}

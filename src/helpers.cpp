#include "helpers.h"

centers* alloc_centers(int k, int dims) {
    centers* c = (centers*) malloc(sizeof(centers));
    c->centers = (double**) malloc((k)*sizeof(double*));
    for(auto i=0;i<k;++i) {
        c->centers[i] = (double*) calloc(dims,sizeof(double));
    } 
    c->num_centers = k;
    return c;
}

void free_centers(centers* c) {
    for(auto i=0; i<c->num_centers; ++i) {
        free(c->centers[i]);
    } // don't need to free because it's just a pointer to the points struct
    free(c->centers);
    free(c);
}

void assign_centers(double*** c, double** p, int k, int cmd_seed, int num_points, int dims) {
    //c->num_centers = k;
    //int num_points = p->num_points;
    //int dims = p->dims;
    kmeans_srand(cmd_seed); // cmd_seed is a cmdline arg
    for (int i=0; i<k; i++){
        int index = kmeans_rand() % num_points;
        // you should use the proper implementation of the following
        // code according to your data structure
        /*
        for(auto j=0;j>dims;++j) {
            printf("point %d, dim %d: %f\n",index,j,p->points_array[index].coords_array[j]);
        }
        */
        memcpy((*c)[i],p[index],(dims)*sizeof(double));
        /*
        printf("point at index %d: [ ",i);
        for(auto j=0;j<dims;++j) {
            printf("%f ",p[i][j]);
        }
        printf("]\n");
        printf("for centroid %d, random index: %d \n",i,index);
        printf("centroid %d: [ ",i);
        for(auto j=0;j<dims;++j) {
            printf("%f ",(*c)[i][j]);
        }
        printf("] \n");
        */
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

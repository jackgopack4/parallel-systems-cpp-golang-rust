#include "helpers.h"

static unsigned long int next = 1;
static unsigned long kmeans_rmax = 32767;

int kmeans_rand() {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next/65536) % (kmeans_rmax+1);
}
void kmeans_srand(unsigned int seed) {
    next = seed;
}
void assign_centers(double** c, double* p, int k, int cmd_seed, int num_points, int dims) {
    kmeans_srand(cmd_seed); // cmd_seed is a cmdline arg
    for (int i=0; i<k; i++){
        int index = kmeans_rand() % num_points;
        memcpy(&(*c)[i*dims],&p[index*dims],(dims)*sizeof(double));
    }
}
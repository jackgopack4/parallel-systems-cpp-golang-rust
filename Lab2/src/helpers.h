#pragma once

#include <stdlib.h>
#include <string.h>
#include <cstdio>

int kmeans_rand();
void kmeans_srand(unsigned int seed);

void assign_centers(double** c, double* p, int k, int cmd_seed, int num_points, int dims);
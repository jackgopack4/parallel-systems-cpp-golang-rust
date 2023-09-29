#ifndef _IO_H
#define _IO_H

#include "argparse.h"
//#include <prefix_sum.h>
#include <iostream>
#include <fstream>
#include "helpers.h"
#include <cstring>
#include <string>
#include <sstream>
#include <cstdio>

void read_file(struct options_t* args,
               double*** points,
               int& num_points);
void print_output(bool clusters, double** p, double** c, int* labels, int num_points, int k, int dims);
#endif
#include <iostream>
#include <stdio.h>
#include "io.h"
#include <chrono>
#include <cstring>
#include "argparse.h"
//#include "operators.h"
#include "helpers.h"
//#include "prefix_sum.h"
//#include "spin_barrier.h"

using namespace std;

int main(int argc, char **argv)
{
    // Parse args
    struct options_t opts;
    get_opts(argc, argv, &opts);
    struct points * input_vals;
    
    read_file(&opts,&input_vals); // allocates input_vals

    free_points(input_vals);
    
    return 0;
}

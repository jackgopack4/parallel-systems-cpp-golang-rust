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

void read_file(struct options_t* args,
               struct points**   input_vals);

#endif
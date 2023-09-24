#pragma once

#include <stdlib.h>
#include <pthread.h>
#include <spin_barrier.h>
#include <iostream>
#include <chrono>

void* compute_prefix_sum(void* a);

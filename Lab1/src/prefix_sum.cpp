#include <iostream>
#include "prefix_sum.h"
#include "helpers.h"
#include <threads.h>
#include <pthread.h>
using namespace std;
void* compute_prefix_sum(void *a)
{
    prefix_sum_args_t *args = (prefix_sum_args_t *)a;
    
    using namespace std::this_thread; // sleep_for, sleep_until
    using namespace std::chrono; // nanoseconds, system_clock, seconds

    /************************
     * Your code here...    *
     * or wherever you like *
     ************************/

    int n_threads = args->n_threads;
    int t_id = args->t_id;
    int n_loops = args->n_loops;
    int pad_length = args->pad_length;
    int (*scan_operator)(int, int, int);
    scan_operator = args->op;
    bool spin = args->spin;

    for (auto stride = 2; stride <= args->pad_length; stride *= 2) {
        auto index = (t_id + 1) * stride - 1;
        /* Used for testing non-sequential thread execution
        if(index <=rand()) {
            sleep_for(nanoseconds(rand()*rand()));
        }
        */
        while(index < pad_length) {
            args->output_vals[index] = scan_operator(args->output_vals[index],args->output_vals[index-stride/2],n_loops);
            index += stride*n_threads;
        }
        if(spin) {
            args->spinbar->barrier_wait(t_id);
        }
        else {
            pthread_barrier_wait((pthread_barrier_t *)args->barrier);
        }
    }


    // downsweep phase
    
    for (int stride = pad_length / 2; stride >1; stride /= 2) {
        auto index = (t_id + 1) * stride - 1;
        /* used for testing non-sequential thread execution
        if(index >rand()) {
            sleep_for(nanoseconds(rand()*rand()));
        }
        */
        while(index+stride/2 < pad_length) {
            args->output_vals[index+stride/2] = scan_operator(args->output_vals[index],args->output_vals[index+stride/2],n_loops);
            index += stride*n_threads;
        }
        if(spin) {
            args->spinbar->barrier_wait(t_id);
        }
        else {
            pthread_barrier_wait((pthread_barrier_t *)args->barrier);
        }
    }
    
    return 0;
}

void* reduction_phase(void *a, pthread_barrier_t *pts) {
    return 0;
}
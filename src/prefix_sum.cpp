#include <iostream>
#include "prefix_sum.h"
#include "helpers.h"
#include <threads.h>
#include <pthread.h>
using namespace std;
void* compute_prefix_sum(void *a)
{
    prefix_sum_args_t *args = (prefix_sum_args_t *)a;
    

    /************************
     * Your code here...    *
     * or wherever you like *
     ************************/
     //
     // start with 3 options: p == 1, bleloch scan on one processor, pad to next 2
     // p < n / 2, group by chunks, do sum of each, calc size of each group, pad so n / p is product of 2
     // p == n / 2, pad and do standard scan
     // p > n / 2, still only start n / 2 processors

     //start_threads(threads, opts.n_threads, ps_args, &function_pointer);
    int n_threads = args->n_threads;
    int t_id = args->t_id;
    int n_loops = args->n_loops;
    //int n_vals = args->n_vals;
    int pad_length = args->pad_length;
    int (*scan_operator)(int, int, int);
    scan_operator = args->op;
    //pthread_barrier_t barrier = args->barrier;
    cout << "t_id = " << t_id << "\n";
    //cout << "barrier: " << &args->barrier << "\n";
    // reduction phase

    cout << "input_vals before scan:\n";
    cout << "{ ";
    for(auto idx=0;idx<pad_length;++idx) {
        cout << args->input_vals[idx] << " ";
    }
    cout << "}\n";
    cout << "output_vals before scan:\n";
    cout << "{ ";
    for(auto idx=0;idx<pad_length;++idx) {
        cout << args->output_vals[idx] << " ";
    }
    cout << "}\n";

    for (auto stride = 2; stride <= args->pad_length; stride *= 2) {
        cout << "stride = " << stride << "\n";
        auto index = t_id*stride+stride-1;
        while(index < pad_length) {
            cout << "index = " << index << "\n";
            args->output_vals[index] = scan_operator(args->output_vals[index],args->output_vals[index-stride/2],n_loops) ;
            index += stride*n_threads;
        }
        cout << "thread " << t_id << " waiting for barrier.\n";
        auto err = pthread_barrier_wait((pthread_barrier_t *)args->barrier);
        cout << "pthread_barrier_wait returned " << err << "\n";
    }
    cout << "output_vals after upsweep:\n";
    cout << "{ ";
    for(auto idx=0;idx<pad_length;++idx) {
        cout << args->output_vals[idx] << " ";
    }
    cout << "}\n";
    //pthread_barrier_wait(&post_reduction);

    // downsweep phase
    /*
    for (int stride = pad_length / 2; stride >0; stride /= 2) {
        pthread_barrier_wait(&pre_reverse);
        int index = (args->t_id+1)*stride*2 - 1;
        if(index+stride < args->n_vals) {
            args->output_vals[index+stride] = args->input_vals[index]+args->output_vals[index+stride];
        }
    }
    */
    return 0;
}

void* reduction_phase(void *a, pthread_barrier_t *pts) {
    return 0;
}

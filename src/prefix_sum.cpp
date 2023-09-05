#include "prefix_sum.h"
#include "helpers.h"
#include <threads.h>
#include <pthread.h>
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
    //int n_loops = args->n_loops;
    //int n_vals = args->n_vals;
    int pad_length = args->pad_length;
    pthread_barrier_t barrier = args->barrier;

    //std::vector<int> inputs(pad_val, 0)
    //std::vector<int> outputs(pad_val,0)
    /*for (int i = 0; i < n_vals; ++i) {
        input[i] = args->input_vals[i];
    }*/
    //std::copy(input_vals,input_vals+n_vals,output_vals);
    // reduction phase
    for (int stride = 2; stride < args->pad_length; stride *= 2) {
        int index = t_id*stride+1;
        while(index < pad_length) {
            args->output_vals[index] += args->output_vals[index-stride/2];
            index += stride*n_threads;
        }
        pthread_barrier_wait(&barrier);
        
    }
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

#include <iostream>
#include <argparse.h>
#include <threads.h>
#include <io.h>
#include <chrono>
#include <cstring>
#include "operators.h"
#include "helpers.h"
#include "prefix_sum.h"
#include <vector>
#include <iterator>

using namespace std;

int main(int argc, char **argv)
{
    // Parse args
    struct options_t opts;
    get_opts(argc, argv, &opts);

    bool sequential = false;
    if (opts.n_threads == 0) {
        opts.n_threads = 1;
        sequential = true;
    }

    // Setup threads
    pthread_t *threads = sequential ? NULL : alloc_threads(opts.n_threads);;

    // Setup args & read input data
    prefix_sum_args_t *ps_args = alloc_args(opts.n_threads);
    int n_vals;
    int *input_vals, *output_vals;
    read_file(&opts, &n_vals, &input_vals, &output_vals);
    //printf((char*)input_vals);
    //auto input_size = std::size(&input_vals);
    cout << "{ ";
    for(auto idx=0;idx<n_vals;++idx) {
        cout << input_vals[idx] << " ";
    }
    cout << "}\n";
    int pad_length = next_power_of_two(n_vals);
    //"op" is the operator you have to use, but you can use "add" to test
    int (*scan_operator)(int, int, int);
    scan_operator = op;
    //scan_operator = add;

    pthread_barrier_t *barrier = alloc_barriers(1);
    fill_args(ps_args, opts.n_threads, n_vals, input_vals, output_vals,
        opts.spin, scan_operator, opts.n_loops, *barrier,pad_length);

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();

    //pthread_barrier_t post_reduction, pre_reverse;

    if (sequential)  {
        //sequential prefix scan
        output_vals[0] = input_vals[0];
        for (auto i = 1; i < n_vals; ++i) {
            //y_i = y_{i-1}  <op>  x_i
            output_vals[i] = scan_operator(output_vals[i-1], input_vals[i], ps_args->n_loops);
        }
    }
    else {
        pthread_barrier_init(barrier, NULL, opts.n_threads);
        //pthread_barrier_init(&pre_reverse, NULL, opts.n_threads);
        start_threads(threads, opts.n_threads, ps_args, &compute_prefix_sum);
        //compute_prefix_sum(ps_args)
        // Wait for threads to finish
        join_threads(threads, opts.n_threads);
        //free(barriers);
    }

    //End timer and print out elapsed
    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "time: " << diff.count() << std::endl;

    // Write output data
    write_file(&opts, &(ps_args[0]));

    // Free other buffers
    free(threads);
    free(ps_args);
    free(barrier);
}

#include "helpers.h"

using namespace std;
prefix_sum_args_t* alloc_args(int n_threads) {
    //cout << "alloc args for n_threads = " << n_threads << "\n";
  return (prefix_sum_args_t*) malloc(n_threads * sizeof(prefix_sum_args_t));
}

int next_power_of_two(int x) {
    int pow = 1;
    while (pow < x) {
        pow *= 2;
    }
    return pow;
}

void fill_args(prefix_sum_args_t *args,
               int n_threads,
               int n_vals,
               int *inputs,
               int *outputs,
               bool spin,
               int (*op)(int, int, int),
               int n_loops,
               pthread_barrier_t *barrier,
               int pad_length) {
    for (auto i = 0; i < n_threads; ++i) {
        args[i] = {inputs, outputs, spin, n_vals, n_threads, i, op, n_loops, barrier, pad_length};
    }
}

pthread_barrier_t* alloc_barriers(int n_barriers) {
    return (pthread_barrier_t*) malloc(n_barriers *sizeof(pthread_barrier_t));
}
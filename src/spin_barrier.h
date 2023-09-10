#ifndef _SPIN_BARRIER_H
#define _SPIN_BARRIER_H

#include <pthread.h>
#include <iostream>
#include <atomic>
/* A tree-based barrier implementation */
struct node {
  int val;
  struct node *left;
  struct node *right;
};

class spin_barrier {
    private:
        node thread_tree;
        std::atomic<bool> *arrive;
        std::atomic<bool> *go;
        std::atomic<bool> is_initialized;
        std::atomic<bool> is_destroyed;
        unsigned thread_count;

    public: 
        spin_barrier();
        ~spin_barrier();
        int barrier_init(unsigned count);
        int barrier_wait();
        int barrier_destroy();
};

#endif

#ifndef _SPIN_BARRIER_H
#define _SPIN_BARRIER_H

#include <pthread.h>
#include <iostream>
#include <atomic>
#include <semaphore.h>
/* A tree-based barrier implementation */
struct node {
  int val;
  struct node *left;
  struct node *right;
};

class spin_barrier {
    private:
        //struct node *thread_tree;
        sem_t *arrive;
        sem_t *go;
        sem_t arrive_counter;
        sem_t go_counter;
        std::atomic<bool> is_initialized;
        std::atomic<bool> is_destroyed;
        int thread_count;

    public: 
        spin_barrier(int count);
        ~spin_barrier();
        int barrier_wait(int t_id);
        //void init_dfs(struct node *cur);
        //void free_dfs(struct node *cur);
};

#endif

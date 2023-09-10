#include <spin_barrier.h>
/* spin_barrier.h contents for easy review:
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
*/

/************************
 * Your code here...    *
 * or wherever you like *
 ************************/

spin_barrier::spin_barrier() {
    is_initialized = false;
    is_destroyed = true;
    thread_count = 0;
}

spin_barrier::~spin_barrier() {
    if(is_initialized && !is_destroyed) {
        // need to check if we are still waiting, if so return error
    }
}

int spin_barrier::barrier_init(unsigned count) {
    return 0;
}

int spin_barrier::barrier_wait() {
    return 0;
}

int spin_barrier::barrier_destroy() {
    // deallocate memory for thread_tree, arrive, and go arrays
    // set thread_count to 0, is_destroyed = true, is_initialized = false
    return 0;
}
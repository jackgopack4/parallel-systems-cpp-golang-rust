#include <spin_barrier.h>
/* spin_barrier.h contents for easy review:
struct node {
  int val;
  struct node *left;
  struct node *right;
};

class spin_barrier {
    private:
        struct node *thread_tree;
        std::atomic<bool> *arrive;
        std::atomic<bool> *go;
        std::atomic<int> arrive_counter;
        std::atomic<int> go_counter;
        std::atomic<bool> is_initialized;
        std::atomic<bool> is_destroyed;
        int thread_count;

    public: 
        spin_barrier(int count);
        ~spin_barrier();
        int barrier_wait();
        void init_dfs(node *cur);
        void free_dfs(node *cur);
};
*/

/************************
 * Your code here...    *
 * or wherever you like *
 ************************/
void spin_barrier::init_dfs(node *cur) {
    auto val = cur->val;
    if (2*val + 1 < thread_count) {
        cur->left = (node*) malloc(sizeof(node));
        cur->left->val = 2*val;
        cur->right = (node*)malloc(sizeof(node));
        cur->right->val = 2*val+1;
        init_dfs(cur->left);
        init_dfs(cur->right);
    }
    else if(2*val < thread_count) {
        cur->left = (node*)malloc(sizeof(node));
        cur->left->val = 2*val;
        cur->right = NULL;
        init_dfs(cur->left);
    }
    else {
        cur->left = NULL;
        cur->right = NULL;
    }
}

void spin_barrier::free_dfs(node *cur) {
    if(cur->left && cur->right) {
        free_dfs(cur->right);
        free_dfs(cur->left);
        free(cur);
    }
    else if(cur->left) {
        free_dfs(cur->left);
        free(cur);
    }
    else {
        free(cur);
    }
}

// must initialize with set number of threads
spin_barrier::spin_barrier(int count) {
    // no error checking currently, must have count larger than 0
    thread_tree = (node*)malloc(sizeof(node));
    thread_tree->val = 0;
    init_dfs(thread_tree);
    arrive = (std::atomic<bool>*)malloc(count*sizeof(std::atomic<bool>));
    go = (std::atomic<bool>*)malloc(count*sizeof(std::atomic<bool>));
    for(auto i=0;i<count;++i) {
        arrive[i]=false;
        go[i]=false;
    }
    arrive_counter= 0;
    go_counter= 0;
    is_initialized = true;
    is_destroyed = false;
    thread_count = count;
}

spin_barrier::~spin_barrier() {
    // todo: add error validation that we can't destroy while still waiting
    free(arrive);
    free(go);
    free_dfs(thread_tree);
}

int spin_barrier::barrier_wait() {
    return 0;
}
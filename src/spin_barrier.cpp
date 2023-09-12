#include <spin_barrier.h>

/************************
 * Your code here...    *
 * or wherever you like *
 ************************/
 /*
void spin_barrier::init_dfs(node *cur) {
    auto val = cur->val;
    if (2*(val+1) + 1 < thread_count) {
        cur->left = (node*) malloc(sizeof(node));
        cur->left->val = 2*(val+1)-1;
        cur->right = (node*)malloc(sizeof(node));
        cur->right->val = 2*(val+1);
        init_dfs(cur->left);
        init_dfs(cur->right);
    }
    else if(2*val < thread_count) {
        cur->left = (node*)malloc(sizeof(node));
        cur->left->val = 2*(val+1)-1;
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
*/
// must initialize with set number of threads
spin_barrier::spin_barrier(int count) {
    // no error checking currently, must have count larger than 0
    //thread_tree = (node*)malloc(sizeof(node));
    //thread_tree->val = 0;
    //init_dfs(thread_tree);
    arrive = (sem_t *)malloc(count*sizeof(sem_t));
    go     = (sem_t *)malloc(count*sizeof(sem_t));
    for(auto i=0;i<count;++i) {
        sem_init(&arrive[i], 0, 0);
        sem_init(&go[i], 0, 0);
    }
    sem_init(&arrive_counter, 0, 0);
    sem_init(&go_counter, 0, 0);
    is_initialized = true;
    is_destroyed = false;
    thread_count = count;
}

spin_barrier::~spin_barrier() {
    // todo: add error validation that we can't destroy while still waiting
    free(arrive);
    free(go);
    //free_dfs(thread_tree);
}

int spin_barrier::barrier_wait(int t_id) {
    if(t_id == 0) {
        if(2<thread_count) {
            sem_wait(&arrive[1]);
            sem_wait(&arrive[2]);
            sem_post(&go[1]);
            sem_post(&go[2]);
        }
        else if(1<thread_count) {
            sem_wait(&arrive[1]);
            sem_post(&go[1]);
        }
    }
    else if((t_id+1) <= (thread_count)/2) {
        if(2*(t_id+1)<thread_count) {
            sem_wait(&arrive[2*(t_id+1)-1]);
            sem_wait(&arrive[2*(t_id+1)]);
            sem_post(&arrive[t_id]);
            sem_wait(&go[t_id]);
            sem_post(&go[2*(t_id+1)-1]);
            sem_post(&go[2*(t_id+1)]);
        }
        else if(2*(t_id+1)-1<thread_count) {
            sem_wait(&arrive[2*(t_id+1)-1]);
            sem_post(&arrive[t_id]);
            sem_wait(&go[t_id]);
            sem_post(&go[2*(t_id+1)-1]);
        }
        else {
            sem_post(&arrive[t_id]);
            sem_wait(&go[t_id]);
        }
    }
    else {
        sem_post(&arrive[t_id]);
        sem_wait(&go[t_id]);
    }
    return 0;
}
#include "io.h"

void read_file(struct options_t* args,
               double*** points,
               int& num_points) {

  	// Open file, count lines

    int n_dims = args->dims;
    std::ifstream in(args->in_file);
    in >> num_points;
    (*points) = (double**) malloc(num_points*sizeof(double*));
    for (int i=0; i < num_points; ++i) {
        (*points)[i] = (double*) calloc(n_dims,sizeof(double));
        std::string in_str;
        std::getline(in, in_str);
        std::stringstream ss(in_str);
        std::string word;
        ss >> word;
        int j = 0;
        while (ss >> word) {
            double tmp_dbl = std::stod(word);
            (*points)[i][j] = tmp_dbl;
            ++j;
        }
    }
}

void print_output(bool clusters, double** p, double** c, int* labels,int num_points, int k, int dims) {
    if(!clusters) {
        printf("clusters:");
        for (int i=0; i < num_points; ++i) {
            printf(" %d", labels[i]);
        }
        printf("\n");
    }
    else {
        for (int i=0;i<k;++i) {
            printf("%d",i);
            for (int j=0;j<dims; ++j) {
                printf(" %f",c[i][j]);
            }
            printf("\n");
        }
    }
}
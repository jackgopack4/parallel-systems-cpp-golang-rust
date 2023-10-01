#include "io.h"

void read_file(struct options_t* args,
               double** points,
               int* num_points) {

  	// Open file, count lines

    int k = args->dims;
    std::string n_string;
    //printf("dim passed to read_file: %d\n",k);
    int n = 0;
    std::ifstream in(args->in_file);
    if (in.is_open()) {
        std::string line;
        if(std::getline(in,line)) {
            try {
                n = std::stoi(line);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Error: Invalid integer in the file." << std::endl;
            } catch (const std::out_of_range& e) {
                std::cerr << "Error: Integer out of range." << std::endl;
            }
        } else {
            std::cerr << "Error: Empty file." << std::endl;
        }
    } else {
        std::cerr << "Error: Unable to open file." << std::endl;
    }
    *num_points = n;
    //printf("num_points: %d\n",n);
    (*points) = (double*) calloc(k*n,sizeof(double));
    for (int i=0; i < n; ++i) {
        std::string in_str;
        std::getline(in, in_str);
        //std::cout << "in_str: " << in_str << "\n";
        std::stringstream ss(in_str);
        std::string word;
        ss >> word;
        int j = 0;
        while (ss >> word) {
            double tmp_dbl = std::stod(word);
            //printf("tmp dbl: %f\n",tmp_dbl);
            (*points)[i*k+j] = tmp_dbl;
            ++j;
        }
    }
}

void print_output(bool clusters, double* p, double* c, int* labels,int num_points, int k, int dims) {
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
                printf(" %f",c[i*dims + j]);
            }
            printf("\n");
        }
    }
}
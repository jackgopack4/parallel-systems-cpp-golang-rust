#include "io.h"

void read_file(struct options_t* args,
               struct points**   input_vals) {

  	// Open file, count lines
    int n_points;

    int n_dims = args->dims;
    std::ifstream in(args->in_file);
    in >> n_points;
    /*
    std::string unused;
    while ( std::getline(in, unused) )
        ++n_points;
    in.clear();
    in.seekg(std::ios::beg);
    */
    *input_vals = alloc_points(n_dims,n_points);
    for (int i=0; i < n_points; ++i) {
        std::string in_str;
        in >> in_str;
        std::cout << "input_str: " << in_str << "\n";
        std::stringstream ss(in_str);
        std::string word;
        ss >> word;
        int j = 0;
        while (ss >> word) {
            (*input_vals)->points_array[i].coords_array[j] = std::stod(word);
            ++j;
        }
    }
}

void print_output(bool clusters, points* p, centers* c, int* labels) {
    int nPoints = p->num_points;
    if(!clusters) {
        printf("clusters:");
        for (int p=0; p < nPoints; p++) {
            printf(" %d", labels[p]);
        }
    }
}
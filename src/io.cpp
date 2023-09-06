#include <io.h>
//#include "helpers.h"

void read_file(struct options_t* args,
               int*              n_vals,
               int**             input_vals,
               int**             output_vals) {

  	// Open file
	std::ifstream in;
	in.open(args->in_file);
	// Get num vals
	in >> *n_vals;
	//std::cout <<"n_vals = " << *n_vals << "\n";
	auto pad_size = next_power_of_two(*n_vals);
	//std::cout << "pad_size = " <<pad_size << " " <<"\n";
	// Alloc input and output arrays
	*input_vals = (int*) malloc(pad_size * sizeof(int));
	*output_vals = (int*) malloc(pad_size * sizeof(int));

	// Read input vals
	for (int i = 0; i < *n_vals; ++i) {
		in >> (*input_vals)[i];
	}
	for (int j=*n_vals; j<pad_size; ++j) {
		(*input_vals)[j] = 0;
	}
}

void write_file(struct options_t*         args,
               	struct prefix_sum_args_t* opts) {
  // Open file
	std::ofstream out;
	out.open(args->out_file, std::ofstream::trunc);

	// Write solution to output file
	for (int i = 0; i < opts->n_vals; ++i) {
		out << opts->output_vals[i] << std::endl;
	}

	out.flush();
	out.close();
	//std::cout << "wrote file, about to free inputs/outputs\n";
	// Free memory
	free(opts->input_vals);
	free(opts->output_vals);
    //free(opts->barrier);
	//std::cout << "freed input and output vals\n";
}

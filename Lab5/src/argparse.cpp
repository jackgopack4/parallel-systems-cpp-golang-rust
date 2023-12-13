#include "argparse.h"

void get_opts(int argc,
              char **argv,
              struct options_t *opts)
{
    if (argc == 1)
    {
        std::cout << "Usage:" << std::endl;
        std::cout << "\t-i <file_path>" << std::endl;
        std::cout << "\t-o <file_path>" << std::endl;
        std::cout << "\t-s <num_steps>" <<std::endl;
        std::cout << "\t-t <theta>" << std::endl;
        std::cout << "\t-d <timestep>" << std::endl;
        std::cout << "\t[Optional] -V <show visualization>" << std::endl;
        std::cout << "\t[Optional] -p <parallel>" << std::endl;
        exit(0);
    }

    opts->steps = 100;
    opts->theta = 0.5;
    opts->dt = 0.005;
    opts->visualize = false;
    opts->parallel = false;
    
    struct option l_opts[] = {
        {"input_file_name", required_argument, NULL, 'i'},
        {"output_file_name", required_argument, NULL, 'o'},
        {"steps", required_argument, NULL, 's'},
        {"theta", required_argument, NULL, 't'},
        {"timestep", required_argument, NULL, 'd'},
        {"visualize", no_argument, NULL, 'v'},
        {"parallel", no_argument, NULL, 'p'},
    };
    int ind, c;
    while ((c = getopt_long(argc, argv, "i:o:s:t:d:vp", l_opts, &ind)) != -1)
    {
        switch (c)
        {
        case 0:
            break;
        case 'i':
            opts->in_file = (char *)optarg;
            break;
        case 'o':
            opts->out_file = (char *)optarg;
            break;
        case 's':
            opts->steps = atoi((char *)optarg);
            break;
        case 't':
            opts->theta = atof((char *)optarg);
            break;
        case 'd':
            opts->dt = atof((char *)optarg);
            break;
        case 'v':
            opts->visualize = true;
            break;
        case 'p':
            opts->parallel = true;
            break;
        case ':':
            std::cerr << argv[0] << ": option -" << (char)optopt << "requires an argument." << std::endl;
            exit(1);
        }
    }
}

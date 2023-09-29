#include "argparse.h"

void get_opts(int argc,
              char **argv,
              struct options_t *opts)
{
    if (argc == 1)
    {
        std::cout << "Usage:" << std::endl;
        std::cout << "\t[Optional] -k <num_cluster>" <<std::endl;
        std::cout << "\t -d <dims>" << std::endl;
        std::cout << "\t-i <file_path>" << std::endl;
        std::cout << "\t[Optional] -m <max_num_iters>" << std::endl;
        std::cout << "\t[Optional] -t <threshold_convergence>" << std::endl;
        std::cout << "\t[Optional] -c" << std::endl;
        std::cout << "\t[Optional] -s <rand_seed>" << std::endl;
        std::cout << "\t[Optional] -v <seq, cuda, shmem, thrust>" << std::endl;
        exit(0);
    }

    opts->num_cluster = 1;
    opts->centroids = false;
    opts->max_num_iter = 1000;
    opts->threshold = 0.02;
    opts->seed = 69;
    opts->version = 0;
    
    struct option l_opts[] = {
        {"num_cluster", no_argument, NULL, 'k'},
        {"dims", required_argument, NULL, 'd'},
        {"input", required_argument, NULL, 'i'},
        {"max_num_iter", no_argument, NULL, 'm'},
        {"threshold", no_argument, NULL, 't'},
        {"centroids", no_argument, NULL, 'c'},
        {"seed", no_argument, NULL, 's'},
        {"version", no_argument, NULL, 'v'},
    };
    
    int ind, c;
    std::string seq{"seq"};
    std::string cuda{"cuda"};
    std::string shmem{"shmem"};
    std::string thrust{"thrust"};
    std::string arg_version;
    while ((c = getopt_long(argc, argv, "k:d:i:m:t:cs:v:", l_opts, &ind)) != -1)
    {
        switch (c)
        {
        case 0:
            break;
        case 'k':
            opts->num_cluster = atoi((char *)optarg);
            break;
        case 'd':
            opts->dims = atoi((char *)optarg);
            break;
        case 'i':
            opts->in_file = (char *)optarg;
            break;
        case 'm':
            opts->max_num_iter = atoi((char *)optarg);
            break;
        case 't':
            opts->threshold = atof((char *)optarg);
            break;
        case 'c':
            opts->centroids = true;
            break;
        case 's':
            opts->seed = atoi((char *)optarg);
            break;
        case 'v':
            arg_version = (char*) optarg;
            if(arg_version.compare(seq) == 0) {
                opts->version = sequential;
            }
            else if (arg_version.compare(cuda) == 0) {
                opts->version = cuda_basic;
            }
            else if (arg_version.compare(shmem) == 0) {
                opts->version = cuda_shmem;
            }
            else if (arg_version.compare(thrust) == 0) {
                opts->version = cuda_thrust;
            }
            break;
        case ':':
            std::cerr << argv[0] << ": option -" << (char)optopt << "requires an argument." << std::endl;
            exit(1);
        }
    }
}

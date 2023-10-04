#include "kmeans.h"

using namespace std;

int main(int argc, char **argv)
{
    // Parse args
    struct options_t opts;
    get_opts(argc, argv, &opts);
    double* points;
    int k = opts.num_cluster;
    int dims = opts.dims;
    int cmd_seed = opts.seed;
    bool cluster_output = opts.centroids;
    int v = opts.version;

    int num_points;
    read_file(&opts,&points,&num_points); // also allocates input_vals
    //printf("read file\n");
    /*
    for(int i=0;i<num_points;++i) {
        printf("point %d: [",i);
        for (int j=0;j<dims; ++j) {
            printf(" %f",points[i*dims + j]);
        }
        printf(" ]\n");
    }
    */
    //printf("num_points = %d\n",num_points);
    double* centroids = (double*) calloc(k*dims, sizeof(double));
    int* indices = (int*) calloc(num_points,sizeof(int));

    assign_centers(&centroids,points,k,cmd_seed, num_points, dims);    
    //printf("assigned centers\n");
    if (v == cuda_basic || v == cuda_shmem || v == cuda_thrust) {
        if (v == cuda_basic) {
            opts.threshold /= 100;
        }
        else {
            opts.threshold /= 1000;
        }
        //printf("threshold = %0.9f\n",opts.threshold);
        auto start = std::chrono::high_resolution_clock::now();
        compute_kmeans_cuda(&opts,points,&centroids,&indices,num_points);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::ratio<1,1000>> diff = end - start;
        std::cout << "total time: " << diff.count() << std::endl;
        print_output(cluster_output,points,centroids,indices,num_points, k, dims);
    }
    
     
    
    free(indices);
    free(centroids);
    free(points);
    return 0;
}

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
                printf(" %0.5f",c[i*dims + j]);
            }
            printf("\n");
        }
    }
}

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
    opts->threshold = 0.00001;
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

static unsigned long int next_idx = 1;
static unsigned long kmeans_rmax = 32767;

int kmeans_rand() {
    next_idx = next_idx * 1103515245 + 12345;
    return (unsigned int)(next_idx/65536) % (kmeans_rmax+1);
}
void kmeans_srand(unsigned int seed) {
    next_idx = seed;
}

void assign_centers(double** c, double* p, int k, int cmd_seed, int num_points, int dims) {
    kmeans_srand(cmd_seed); // cmd_seed is a cmdline arg
    for (int i=0; i<k; i++){
        int index = kmeans_rand() % num_points;
        memcpy(&(*c)[i*dims],&p[index*dims],(dims)*sizeof(double));
    }
}
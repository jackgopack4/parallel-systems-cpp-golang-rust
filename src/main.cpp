/*
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <GL/glu.h>
#include <GL/glut.h>
*/
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <sys/resource.h>
#include <time.h>
#include <iomanip>
#include <unistd.h>
//#include "datavector.h"
#include "body.h"
#include "bodyfilereader.h"
#include "bodyfilewriter.h"
#include "argparse.h"
#include "quad.h"
#include "bhtree.h"
#include <mpi.h>

using namespace std;



int main(int argc, char **argv) {
  cout << fixed << setprecision(6);
  /*
  const rlim_t kStackSize = 256 * 1024 * 1024;   // min stack size = 16 MB
  struct rlimit rl;
  int result;
  result = getrlimit(RLIMIT_STACK, &rl);
  if (result == 0)
  {
    if (rl.rlim_cur < kStackSize)
    {
      rl.rlim_cur = kStackSize;
      result = setrlimit(RLIMIT_STACK, &rl);
      if (result != 0)
      {
        cerr << "setrlimit returned result = " << result << endl;
        //fprintf(stderr, "setrlimit returned result = %d\n", result);
      }
    }
  }
  */

  // Initialize MPI
  MPI_Init(&argc, &argv);

  // Parse args
  struct options_t opts;
  get_opts(argc, argv, &opts);

  const std::string in_file(opts.in_file);
  const std::string out_file(opts.out_file);
  //int num_steps = opts.steps;
  //double theta = opts.theta;
  //double timestep = opts.dt;
  //bool visualize = opts.visualize;
  //bool parallel = opts.parallel;

  BodyFileReader bodyReader(in_file);
  vector<Body> bodies = bodyReader.readBodies();

  // Get the rank and size of the MPI communicator
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  //int num_bodies = bodies.size();
  //int local_num_bodies = num_bodies / size;
  //int remaining_bodies = num_bodies % size;

  //int num_bodies{(int) bodies.size()};
  // record the start time
  clock_t start_time = clock();
  for (auto i = 0; i < opts.steps; ++i)
  {
    clock_t lap_start_time = clock();
    //cout << "starting run " << i << endl;
    vector<Datavector> forces(bodies.size());
    //cout << "allocated new forces datavector" << endl;
    Quad test_quad(0.0,0.0,4.0);
    BHTree* test_tree = new BHTree(test_quad);
    //cout << "allocated new test_tree" << endl;
    // Steps for the loop - 
    // 1. build tree
    // 2. update force vector by iterating through tree
    // 3. apply position and velocity updates using leapfrog vertlet
    // repeat for num_steps
    
    // create the tree by inserting all bodies
    for(Body b: bodies) {
      test_tree->insert(b);
    }

    // calculate force on each body by traversing the tree
    for(Body b: bodies) {
      int idx = b.index;
      if (b.mass > -0.0000000001) {
        double x_comp = 0.0;
        double y_comp = 0.0;
        if (test_tree->body.mass > 0.0000000001) test_tree->calculateForce(b,opts.theta,&x_comp,&y_comp);
        forces[idx].data[0] = x_comp;
        forces[idx].data[1] = y_comp;// = *new_force;
      }
      
    }

    // apply movement to each body for given timestep
    for(auto i=0;i<(int)bodies.size();++i) {
      bodies[i].move(&forces[i],opts.dt);
    }
    delete test_tree;
    forces.clear();
    clock_t lap_time = clock();
    double elapsed_lap_time = (double)(lap_time - lap_start_time) / CLOCKS_PER_SEC;
    //cout << "Elapsed time for cycle "<< i <<": " << elapsed_lap_time << "seconds\n";
    cout << elapsed_lap_time << endl;
    //usleep(500000);
  }

  // Finalize MPI
  MPI_Finalize();
  // Record the ending time
  clock_t end_time = clock();

  // Calculate the elapsed time in seconds
  double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;// - (0.5*opts.steps);

  // Print the elapsed time
  //cout << "Elapsed time: " << elapsed_time << " seconds\n";
  cout << elapsed_time << endl;

  //test_tree->updateVectorWithBodies(bodies);
  BodyFileWriter bodyWriter(out_file);
  bodyWriter.writeBodies(bodies);

}
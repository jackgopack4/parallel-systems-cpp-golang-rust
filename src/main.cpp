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

struct forceStruct {
    int idx;
    double x_comp;
    double y_comp;
};
MPI_Datatype createRecType()
{
    // Set-up the arguments for the type constructor
    MPI_Datatype new_type;

    int count = 3;
    int blocklens[] = { 1,1,1 };

    MPI_Aint indices[3];
    indices[0] = (MPI_Aint)offsetof(struct forceStruct, idx);
    indices[1] = (MPI_Aint)offsetof(struct forceStruct, x_comp);
    indices[2] = (MPI_Aint)offsetof(struct forceStruct, y_comp);

    MPI_Datatype old_types[] = {MPI_INT,MPI_DOUBLE,MPI_DOUBLE};

    MPI_Type_struct(count,blocklens,indices,old_types,&new_type);
    MPI_Type_commit(&new_type);

    return new_type;
}

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

  if (size > 1) {
    // Parallel version
    int num_bodies = bodies.size();

    auto forceStructType = createRecType();
    MPI_Type_commit(&forceStructType);
    // Record the start time
    clock_t start_time = clock();
    double startwtime = 0.0, endwtime;
    if (rank == 0)
        startwtime = MPI_Wtime();

    
    for (auto i = 0; i < opts.steps; ++i) {
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
      MPI_Barrier(MPI_COMM_WORLD);
      // calculate force on each local body by traversing the tree
      for(int j=0;j<num_bodies;++j) {
        forceStruct data;
        int idx = bodies[j].index;
        if(bodies[j].mass > 0.0000000001) {
          if( ( (j +size - rank) % size) == 0 && rank < num_bodies ) {
            // need to do the math and broadcast
            double x_comp = 0.0;
            double y_comp = 0.0;
            if (test_tree->body.mass > 0.0)
              test_tree->calculateForce(bodies[j],opts.theta,&x_comp,&y_comp);
            forces[idx].data[0] = x_comp;
            forces[idx].data[1] = y_comp;
            data.idx = idx;
            data.x_comp = x_comp;
            data.y_comp = y_comp;
            MPI_Bcast(&data,1,forceStructType,rank,MPI_COMM_WORLD);
            //cout << "process rank " << rank << ", idx " << j << ", bcasted data\n";
          } else {
            int bcast_node = rank-((j+size-rank)%size);
            if(bcast_node < 0) bcast_node += size;
            MPI_Bcast(&data,1,forceStructType,bcast_node,MPI_COMM_WORLD);
            //cout << "process rank " << rank << ", idx " << j << ", rx'd data\n";
            forces[data.idx].data[0] = data.x_comp;
            forces[data.idx].data[1] = data.y_comp;
          }
        }

      }
      MPI_Barrier(MPI_COMM_WORLD);

      // apply movement to each body for given timestep
      for(auto i=0;i<(int)bodies.size();++i) {
        bodies[i].move(&forces[i],opts.dt);
      }
      delete test_tree;
      forces.clear();
      clock_t lap_time = clock();
      double elapsed_lap_time = (double)(lap_time - lap_start_time) / CLOCKS_PER_SEC;
      //cout << "Elapsed time for cycle "<< i <<": " << elapsed_lap_time << "seconds\n";
      cout << "process rank " << rank << "lap time (s): " << elapsed_lap_time << endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) {
      endwtime = MPI_Wtime();
      printf("rank 0 total time = %f\n", endwtime - startwtime);
    }
    clock_t end_time = clock();

    // Calculate the elapsed time in seconds
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;// - (0.5*opts.steps);

    cout << "process rank " << rank << "time (s): " << elapsed_time << endl;

  } 
  else {
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
    // Record the ending time
    clock_t end_time = clock();

    // Calculate the elapsed time in seconds
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;// - (0.5*opts.steps);

    // Print the elapsed time
    //cout << "Elapsed time: " << elapsed_time << " seconds\n";
    cout <<"sequential time (s): " << elapsed_time << endl;
  }

  
  

  // Finalize MPI
  MPI_Finalize();

  //test_tree->updateVectorWithBodies(bodies);
  if(rank == 0) {
    BodyFileWriter bodyWriter(out_file);
    bodyWriter.writeBodies(bodies);
  }
  return 0;
}
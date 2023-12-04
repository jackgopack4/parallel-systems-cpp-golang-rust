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
#include <memory>

using namespace std;

struct forceStruct {
    int bodyIdx;
    double x_comp;
    double y_comp;
    forceStruct(): bodyIdx(-1),x_comp(0.0),y_comp(0.0) {}
};
MPI_Datatype createRecType()
{
    // Set-up the arguments for the type constructor
    MPI_Datatype new_type;

    int count = 3;
    int blocklens[] = { 1,1,1 };

    MPI_Aint indices[3];
    indices[0] = (MPI_Aint)offsetof(struct forceStruct, bodyIdx);
    indices[1] = (MPI_Aint)offsetof(struct forceStruct, x_comp);
    indices[2] = (MPI_Aint)offsetof(struct forceStruct, y_comp);

    MPI_Datatype old_types[] = {MPI_INT,MPI_DOUBLE,MPI_DOUBLE};
 //MPI_Type_create_struct(3, blocklen, disp, type, &Particletype);
    MPI_Type_create_struct(count,blocklens,indices,old_types,&new_type);
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
    int local_num_bodies = num_bodies / size;
    int remaining_bodies = num_bodies % size;

    // Calculate local offset and count
    int local_offset = rank * local_num_bodies + min(rank, remaining_bodies);
    int local_count = local_num_bodies + (rank < remaining_bodies ? 1 : 0);
    const int* recvCountPtr;
    const int* displsPtr;
    const void* dataPtr;
    int recvCounts[size];
    int displs[size];
    for(int i = 0;i < size; ++i) {
      recvCounts[i] = local_num_bodies + (i < remaining_bodies ? 1 : 0);
      displs[i] = i * local_num_bodies + min(i, remaining_bodies);
    }
    recvCountPtr = (const int*)&recvCounts;
    displsPtr = (const int*)&displs;
    //cout << "local offset for node["<<rank<<"] = " << local_offset << endl;

    auto forceStructType = createRecType();
    MPI_Type_commit(&forceStructType);
    MPI_Barrier(MPI_COMM_WORLD);
    // Record the start time
    //clock_t start_time = clock();
    double startwtime = 0.0, endwtime;
    if (rank == 0)
      startwtime = MPI_Wtime();

    
    for (auto i = 0; i < opts.steps; ++i) {
      //clock_t lap_start_time = clock();
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
      //MPI_Barrier(MPI_COMM_WORLD);
      // calculate force on each local body by traversing the tree
      /*
      int local_count = 0;
      for(int j=0;j<num_bodies;++j) {
        if ((j % size) == rank) local_count++;
      }
      */
      int my_data_count = 0;
      forceStruct* dataTx = new forceStruct[local_count];
      dataPtr = (const void*)dataTx;
      //std::vector<forceStruct> dataVec(local_count+1);
      //forceStruct* dataPtr = data;
      for(int j=0;j<num_bodies;++j) {
        int idx = bodies[j].index;
        //int bcast_node_rank = j % size;
        if( j >= local_offset && j < (local_offset+local_count)) {
          // need to do the math and broadcast
          //cout << "process rank " << rank << ", bcast, idx " << j <<"\n";
          //cout << "node "<< rank << " calculating force body idx[" << idx <<"]\n";
          double x_comp = 0.0;
          double y_comp = 0.0;
          if (test_tree->body.mass > 0.0)
            test_tree->calculateForce(bodies[j],opts.theta,&x_comp,&y_comp);
          //forces[idx].data[0] = x_comp;
          //forces[idx].data[1] = y_comp;
          if(rank == 0) {
            //cout << "node " << rank << " force "<<idx<<" after initial calc: " << forces[idx] << endl;
          }
          dataTx[my_data_count].bodyIdx = idx;
          dataTx[my_data_count].x_comp = x_comp;
          dataTx[my_data_count].y_comp = y_comp;
          //cout <<"node "<<rank<<" loaded ["<< dataTx[my_data_count].bodyIdx << ", " << dataTx[my_data_count].x_comp << ", "<< dataTx[my_data_count].x_comp << "]" << endl;
          my_data_count++;
          //data.idx = idx;
          //data.x_comp = x_comp;
          //data.y_comp = y_comp;
          //MPI_Bcast(&data,1,forceStructType,bcast_node_rank,MPI_COMM_WORLD);
          //cout << "process rank " << rank << ", idx " << j <<" bcast success\n";
          //cout << "process rank " << rank << ", idx " << j << ", bcasted data\n";
        }

        /*
        else {
          //cout << "process rank " << rank << ", rx, idx " << j << "\n";
          //int bcast_node = rank-((j+size-rank)%size);
          //if(bcast_node < 0) bcast_node += size;
          //MPI_Bcast(&data,1,forceStructType,bcast_node_rank,MPI_COMM_WORLD);
          //cout << "process rank " << rank << ", idx " << data.idx << " rx success\n";
          //cout << "process rank " << rank << ", idx " << j << ", rx'd data\n";
          //forces[data.idx].data[0] = data.x_comp;
          //forces[data.idx].data[1] = data.y_comp;
        }
        */
        // Now that data has been loaded for local nodes, need to loop through and broadcast it
        
      }
      forceStruct* dataRx = new forceStruct[num_bodies];
      //cout << "Node[" << rank << "] abt to tx following data:\n";
      for(int j=0;j<local_count;++j) {
        //dataRx[j+local_offset].bodyIdx = dataTx[j].bodyIdx;
        //dataRx[j+local_offset].x_comp = dataTx[j].x_comp;
        //dataRx[j+local_offset].y_comp = dataTx[j].y_comp;
        //cout << dataTx[j].bodyIdx << ", " << dataTx[j].x_comp << ", " << dataTx[j].y_comp << endl;
      }
      //MPI_Barrier(MPI_COMM_WORLD);
      MPI_Allgatherv(
        dataPtr,
        local_count,
        forceStructType,
        dataRx,
        recvCountPtr,
        displsPtr,
        forceStructType,
        MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);
      for(int j=0;j<num_bodies;++j) {
        //cout << "after allgatherv rx on node "<< rank <<": idx["<<j<<"]: [" << dataRx[j].bodyIdx << ", " << dataRx[j].x_comp << ", " << dataRx[j].y_comp << "]\n";
        int idx = dataRx[j].bodyIdx;
        double x_comp = dataRx[j].x_comp;
        double y_comp = dataRx[j].y_comp;
        //forces[idx] = Datavector({x_comp,y_comp});
        forces[idx].data[0] = x_comp;
        forces[idx].data[1] = y_comp;
      }
      /*
      for(int j=0;j<size;++j) {
        int new_local_count;
        if(j == rank) {
          new_local_count = local_count;
        }
        else {
          new_local_count = local_num_bodies + (j < remaining_bodies ? 1 : 0);
        }
        if(new_local_count > 0) {
          //MPI_Barrier(MPI_COMM_WORLD);
          //MPI_Request request_x;
          //int data_idx_1 = data[0].bodyIdx;
          //MPI_Bcast(data,new_local_count,forceStructType,j,MPI_COMM_WORLD);
          //MPI_Barrier(MPI_COMM_WORLD);
          if(j != rank) {
            
            for(int k=0;k<new_local_count;++k) {
              int force_idx = data[k].bodyIdx;
              double x_comp = data[k].x_comp;
              double y_comp = data[k].y_comp;
              forces[force_idx].data[0] = x_comp;
              forces[force_idx].data[1] = y_comp;
              cout << "rx data fm node "<< j <<": [" <<force_idx << ", " << x_comp <<", " << y_comp << "]" <<endl;
            }
          }
        }
      }
      */
      MPI_Barrier(MPI_COMM_WORLD);
      // apply movement to each body for given timestep
      for(auto i=0;i<(int)bodies.size();++i) {
        if(rank == 0)

          //cout <<"node 0 updating force[" << i << "]: " << forces[i] << endl;
        bodies[i].move(&forces[i],opts.dt);
      }
      delete test_tree;
      delete[] dataRx;
      delete[] dataTx;
      forces.clear();
      //clock_t lap_time = clock();
      //double elapsed_lap_time = (double)(lap_time - lap_start_time) / CLOCKS_PER_SEC;
      //cout << "Elapsed time for cycle "<< i <<": " << elapsed_lap_time << "seconds\n";
      MPI_Barrier(MPI_COMM_WORLD);
      
      if(rank == 0 && num_bodies > 10000) {
        //cout << elapsed_lap_time << endl;
      }
      
      //delete data;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) {
      endwtime = MPI_Wtime();
      cout << (endwtime-startwtime) << endl;
      //printf("rank 0 total time = %f\n", endwtime - startwtime);
    }
    MPI_Type_free(&forceStructType);
    //clock_t end_time = clock();

    // Calculate the elapsed time in seconds
    //double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;// - (0.5*opts.steps);

    //cout << "process rank " << rank << "time (s): " << elapsed_time << endl;

  } 
  else {
    // record the start time
    clock_t start_time = clock();
    for (auto i = 0; i < opts.steps; ++i)
    {
      //clock_t lap_start_time = clock();
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
      //clock_t lap_time = clock();
      //double elapsed_lap_time = (double)(lap_time - lap_start_time) / CLOCKS_PER_SEC;
      //cout << "Elapsed time for cycle "<< i <<": " << elapsed_lap_time << "seconds\n";
      //cout << elapsed_lap_time << endl;
      //usleep(500000);
      if(bodies.size() > 10000) {
        BodyFileWriter bodyWriter(out_file);
        bodyWriter.writeBodies(bodies);
      }
    }
    // Record the ending time
    clock_t end_time = clock();

    // Calculate the elapsed time in seconds
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;// - (0.5*opts.steps);

    // Print the elapsed time
    //cout << "Elapsed time: " << elapsed_time << " seconds\n";
    cout << elapsed_time << endl;
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
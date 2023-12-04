#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <GL/glu.h>
#include <GL/glut.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
//#include "datavector.h"
#include "body.h"
#include "bodyfilereader.h"
#include "bodyfilewriter.h"
#include "argparse.h"
#include "quad.h"
#include "bhtree.h"

using namespace std;



int main(int argc, char **argv) {
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
  for(auto b:bodies) {
    cout << b << endl;
  }
  //int num_bodies{(int) bodies.size()};
  for (auto i = 0; i < opts.steps; ++i)
  {
    //cout<< "starting run " << i << endl;
    //vector<Datavector> forces(bodies.size());
    vector<double*> force_arrs;
    for(auto i=0;i<(int)bodies.size();++i) {
      double* tmp_force = new double[2]{0.0, 0.0};

        // Add the array to the vector
      force_arrs.push_back(tmp_force);
    }
    /*
    for(auto f: force_arrs) {
      cout <<"pointer" << (void *)f << endl;
    }
    */
    //cout << "datavec size: " << sizeof(forces[0]) << endl;
    //cout << "allocated new forces datavector" << endl;
    Quad test_quad(0.0,0.0,4.0);
    BHTree* test_tree = new BHTree(test_quad);
    //cout << "allocated new test_tree" << endl;
    // Steps for the loop - 
    // 1. build tree
    // 2. update force vector by iterating through tree
    // 3. apply position and velocity updates using leapfrog vertlet
    // repeat for num_steps
    

    for(Body b: bodies) {
      /*
      cout << b;
      if (test_quad.contains(b)) {
        cout << ", body in quad";
      }
      cout << endl;
      */
      test_tree->insert(b);
      //cout << test_tree << endl;
    }
    cout << *test_tree << endl;
    //cout << "finished tree build\n";
    for(auto j=0;j<(int)bodies.size();++j) {
      //if(j % 10000 == 0) {
        //cout << "calculating force for idx " << j <<endl;
      //}
      if (bodies[j].mass > 0.0) {
        test_tree->calculateForceArr(bodies[j],opts.theta,force_arrs[j]);
      }
    }
    /*
    for(Body b: bodies) {
      int idx = b.getIndex();
      if(idx%1000 == 0) {
        cout << "calculating force for idx " << idx <<endl;
      }
      if (b.getMass() > 0.0) {
        //cout << "updating mass for body idx " << idx << endl;
        //Datavector new_force = test_tree.calculateForce(b,opts.theta);
        test_tree.calculateForceArr(b,opts.theta,&(force_arrs[idx]));
        //forces[idx] = new_force;
        //cout << "updated mass idx " << idx << endl;
      }
      
    }
    */
    //cout << "finished force calculations\n";
    for(auto j=0;j<(int)bodies.size();++j) {
      //int idx = bodies[i].getIndex();
      //Body b = bodies[i];
      //cout << "attempting to move mass body idx " << idx << endl;
      //bodies[j].move(force_arrs[j],opts.dt);
      positionAndVelocityUpdate(&bodies[j],force_arrs[j],opts.dt);
      //delete &forces[j];
      //cout << "moved mass body idx " << idx << endl;
    }

    for(auto f: force_arrs) {
      delete f;
    }
    delete test_tree;
    /*
    vector<Body> new_bodies(bodies.size());
    int len_bodies{(int)bodies.size()};
    for(int i{0};i<len_bodies;++i) 
    {
      if((bodies[i] != new_bodies[i]))
      {
        cout << "idx [" << i <<"] not equal: " << bodies[i] << " != " << new_bodies[i] << endl;
      } else {
        cout << "idx[" << i <<"] matches!" <<endl;
      }
    }
    */

  }

  BodyFileWriter bodyWriter(out_file);
  bodyWriter.writeBodies(bodies);

}
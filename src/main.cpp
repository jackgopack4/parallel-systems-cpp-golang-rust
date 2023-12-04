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
#include <sys/resource.h>
//#include "datavector.h"
#include "body.h"
#include "bodyfilereader.h"
#include "bodyfilewriter.h"
#include "argparse.h"
#include "quad.h"
#include "bhtree.h"

using namespace std;



int main(int argc, char **argv) {

const rlim_t kStackSize = 128 * 1024 * 1024;   // min stack size = 16 MB
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
        fprintf(stderr, "setrlimit returned result = %d\n", result);
    }
  }
}


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
  //int num_bodies{(int) bodies.size()};
  for (auto i = 0; i < opts.steps; ++i)
  {
    cout << "starting run " << i << endl;
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
    //cout << test_tree << endl;

    for(Body b: bodies) {
      int idx = b.index;
      if (b.mass > -0.0000000001) {
        //cout << "updating mass for body idx " << idx << endl;
        double x_comp = 0.0;
        double y_comp = 0.0;
        if (test_tree->body.mass > 0.0000000001) test_tree->calculateForce(b,opts.theta,&x_comp,&y_comp);
        forces[idx].data[0] = x_comp;
        forces[idx].data[1] = y_comp;// = *new_force;
        //delete new_force;
        //cout << "updated mass idx " << idx << endl;
      }
      
    }

    for(auto i=0;i<(int)bodies.size();++i) {
      //int idx = bodies[i].getIndex();
      //Body b = bodies[i];
      //cout << "attempting to move mass body idx " << idx << endl;
      bodies[i].move(&forces[i],opts.dt);
      //cout << "moved mass body idx " << idx << endl;
    }
    /*
    for(auto f: forces) {
      delete forces[idx];
    }
    */
    /*
    vector<Body> new_bodies(bodies.size());
    test_tree->updateVectorWithBodies(new_bodies);
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
    delete test_tree;
  }

  //test_tree->updateVectorWithBodies(bodies);
  BodyFileWriter bodyWriter(out_file);
  bodyWriter.writeBodies(bodies);

}
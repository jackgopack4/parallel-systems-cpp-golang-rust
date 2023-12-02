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


  BodyFileReader bodyReader(in_file);
  vector<Body> bodies = bodyReader.readBodies();

  Quad test_quad(0.0,0.0,4.0);
  BHTree test_tree(test_quad);
  for(Body b: bodies) {
    /*
    cout << b;
    if (test_quad.contains(b)) {
      cout << ", body in quad";
    }
    cout << endl;
    */
    test_tree.insert(b);
    //cout << test_tree << endl;
  }
  cout << test_tree << endl;
  /*
  vector<Body> new_bodies(bodies.size());
  test_tree.updateVectorWithBodies(new_bodies);
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
  test_tree.updateVectorWithBodies(bodies);
  BodyFileWriter bodyWriter(out_file);
  bodyWriter.writeBodies(bodies);

}
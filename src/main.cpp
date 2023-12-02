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

using namespace std;

int main(int argc, char **argv) {
  // Parse args
  struct options_t opts;
  get_opts(argc, argv, &opts);

  const std::string in_file(opts.in_file);
  const std::string out_file(opts.out_file);


  BodyFileReader bodyReader(in_file);
  vector<Body> bodies = bodyReader.readBodies();

  for(Body b: bodies) {
    cout << b << endl;
  }

  BodyFileWriter bodyWriter(out_file);
  bodyWriter.writeBodies(bodies);

}
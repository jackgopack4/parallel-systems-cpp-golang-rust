#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <GL/glu.h>
#include <GL/glut.h>
#include <iostream>
#include <vector>
//#include "datavector.h"
#include "body.h"
#include "bodyfilereader.h"

int main() {
  BodyFileReader bodyReader("input/nb-10.txt");
  std::vector<Body> bodies = bodyReader.readBodies();

  for(Body b: bodies) {
    std::cout << b << std::endl;
  }
}
#ifndef BHTREE_H
#define BHTREE_H
#include <vector>
#include <iostream>
#include <string>
#include "quad.h"
#include <string_view>
#include <cmath>

class BHTree {
private:
  
  void split();
  void moveExistingBody();
  void printTree(int indent, std::ostream &os, std::string_view quadrant);
  void updateAggregateBody();
public:
  Body body;     // body or aggregate body stored in this node
  Quad quad;     // square region that the tree represents
  BHTree* NW;     // tree representing northwest quadrant
  BHTree* NE;     // tree representing northeast quadrant
  BHTree* SW;     // tree representing southwest quadrant
  BHTree* SE;     // tree representing southeast quadrant
  // Constructor
  BHTree(Body& _body, Quad& _quad, BHTree& _NW, BHTree& _NE, BHTree& _SW, BHTree& _SE);
  BHTree(Quad& _quad);
  BHTree();
  ~BHTree();
  // methods
  void insert(Body& b);
  void insertIntoQuadrant(Body& b);
  void updateVectorWithBodies(std::vector<Body>& bodies);
  void calculateForce(Body& b, double theta, double* x_comp, double* y_comp);
  friend std::ostream& operator << (std::ostream &os, BHTree &bht)
  {
    bht.printTree(0,os,"root");
    return os;
  }
};
void forceFrom(double* x_comp, double* y_comp, Body& b0, Body& b1);
void calculateForceHelper(BHTree* bht, Body& b1, double theta, double* x_comp, double* y_comp);
#endif // BHTREE_H
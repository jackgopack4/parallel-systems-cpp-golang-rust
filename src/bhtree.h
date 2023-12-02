#ifndef BHTREE_H
#define BHTREE_H

#include "quad.h"

class BHTree {
private:
  Body body;     // body or aggregate body stored in this node
  Quad quad;     // square region that the tree represents
  BHTree* NW;     // tree representing northwest quadrant
  BHTree* NE;     // tree representing northeast quadrant
  BHTree* SW;     // tree representing southwest quadrant
  BHTree* SE;     // tree representing southeast quadrant
  void split();
  void moveExistingBody();
  void printTree(int indent, std::ostream &os);

public:
  // Constructor
  BHTree(Body& _body, Quad& _quad, BHTree& _NW, BHTree& _NE, BHTree& _SW, BHTree& _SE);
  BHTree(Quad& _quad);
  BHTree();

  // methods
  void insert(Body& b);
  void insertIntoQuadrant(Body& b);
  void updateVectorWithBodies(std::vector<Body>& bodies);
  friend std::ostream& operator << (std::ostream &os, BHTree &bht)
  {
    bht.printTree(0,os);
    return os;
  }
};

#endif // BHTREE_H

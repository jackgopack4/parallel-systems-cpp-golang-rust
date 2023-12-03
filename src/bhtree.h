#ifndef BHTREE_H
#define BHTREE_H
#include <vector>
#include <iostream>
#include <string>
#include "quad.h"
#include <string_view>
#include <memory>
class BHTree: public std::enable_shared_from_this<BHTree> {
private:
  Body body;     // body or aggregate body stored in this node
  Quad quad;     // square region that the tree represents
  std::shared_ptr<BHTree> NW;     // tree representing northwest quadrant
  std::shared_ptr<BHTree> NE;     // tree representing northeast quadrant
  std::shared_ptr<BHTree> SW;     // tree representing southwest quadrant
  std::shared_ptr<BHTree> SE;     // tree representing southeast quadrant
  void split();
  void moveExistingBody();
  void printTree(int indent, std::ostream &os, std::string_view quadrant);
  void updateAggregateBody();
public:
  // Constructor
  BHTree(Body& _body, Quad& _quad, BHTree& _NW, BHTree& _NE, BHTree& _SW, BHTree& _SE);
  BHTree(Quad& _quad);
  BHTree();
  //~BHTree();
  // methods
  //void DestroyRecursive(std::shared_ptr<BHTree> ptr);
  void insert(Body& b);
  void insertIntoQuadrant(Body& b);
  void updateVectorWithBodies(std::vector<Body>& bodies);
  Datavector calculateForce(Body& b, double theta);
  friend std::ostream& operator << (std::ostream &os, BHTree &bht)
  {
    bht.printTree(0,os,"root");
    return os;
  }
};

#endif // BHTREE_H

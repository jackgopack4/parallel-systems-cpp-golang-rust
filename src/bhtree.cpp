#include "bhtree.h"

BHTree::BHTree(Body& _body, Quad& _quad, BHTree& _NW, BHTree& _NE, BHTree& _SW, BHTree& _SE)
  : body(_body), quad(_quad), NW(&_NW), NE(&_NE), SW(&_SW), SE(&_SE) {}

BHTree::BHTree(Quad& _quad) : body(), quad(_quad), NW(nullptr), NE(nullptr), SW(nullptr), SE(nullptr) {}

BHTree::BHTree() : body(), quad(Quad(0.0, 0.0, 4.0)), NW(nullptr), NE(nullptr), SW(nullptr), SE(nullptr) {}
/*
void BHTree::DestroyRecursive(std::shared_ptr<BHTree> ptr)
{
  if (ptr)
  {
    DestroyRecursive((*ptr).NW);
    DestroyRecursive((*ptr).NE);
    DestroyRecursive((*ptr).SW);
    DestroyRecursive((*ptr).SE);
    ptr = nullptr;
  }
}

BHTree::~BHTree()
{
    DestroyRecursive(shared_from_this());
}
*/
/*
BHTree::~BHTree()
{
  if(NW == nullptr)
  {
    delete &body;
    delete &quad;
    return;
  }
  delete NW;
  delete NE;
  delete SE;
  delete SW;
}*/

void BHTree::insert(Body& b)
{
  // If the node is empty, insert the body here
  if (body.mass == 0.0) 
  {
    body = b;
    return;
  }
  if(NW == nullptr)
  {
    split();
  }
  if (!body.checkAggregate()) {
    moveExistingBody();
  }
  insertIntoQuadrant(b);
}

void BHTree::split()
{
  double newWidth = quad.width / 2.0;
  double newHeight = quad.height / 2.0;
  double curMinX = quad.minX;
  double curMinY = quad.minY;
  Quad sw_quad(curMinX,curMinY,newWidth,newHeight);
  SW = std::make_shared<BHTree>(sw_quad);
  Quad nw_quad(curMinX,curMinY+newHeight,newWidth,newHeight);
  NW = std::make_shared<BHTree>(nw_quad);
  Quad se_quad(curMinX+newWidth,curMinY,newWidth,newHeight);
  SE = std::make_shared<BHTree>(se_quad);
  Quad ne_quad(curMinX+newWidth,curMinY+newHeight,newWidth,newHeight);
  NE = std::make_shared<BHTree>(ne_quad);
  //moveExistingBody();
  //insertIntoQuadrant(body);

}

void BHTree::updateAggregateBody()
{
  double totalMass = (*NW).body.mass + (*NE).body.mass + (*SW).body.mass + (*SE).body.mass;

  if (totalMass > 0.0) {
    double weightedX = (*NW).body.position.data[0] * (*NW).body.mass + (*NE).body.position.data[0] * (*NE).body.mass
                      + (*SW).body.position.data[0] * (*SW).body.mass + (*SE).body.position.data[0] * (*SE).body.mass;

    double weightedY = (*NW).body.position.data[1] * (*NW).body.mass + (*NE).body.position.data[1] * (*NE).body.mass
                      + (*SW).body.position.data[1] * (*SW).body.mass + (*SE).body.position.data[1] * (*SE).body.mass;

    double centerX = weightedX / totalMass;
    double centerY = weightedY / totalMass;

    // Update the aggregate body information in this node
    body.setPosition({centerX,centerY});
    body.setMass(totalMass);
    body.makeAggregate();
  }
}

// Helper function to move the existing body to the appropriate quadrant and update aggregate information
void BHTree::moveExistingBody() {
  if (NW == nullptr) {
    // This is a leaf node, no need to move the existing body
    return;
  }
  insertIntoQuadrant(body);

  // Calculate the aggregate body information


    //isAggregate = true;
    //body(-1,{centerX,centerY},totalMass,true);
    //body.position.x = centerX;
    //body.position.y = centerY;
    //body.setMass(totalMass);
}


// Helper function to insert a body into the appropriate quadrant
void BHTree::insertIntoQuadrant(Body& b) 
{
  double midX = quad.getMinX() + quad.width / 2.0;
  double midY = quad.getMinY() + quad.height / 2.0;

  if (b.position.data[0] <= midX) {
    if (b.position.data[1] <= midY) {
      (*SW).insert(b);
    } else {
      (*NW).insert(b);
    }
  } else {
    if (b.position.data[1] <= midY) {
      (*SE).insert(b);
    } else {
      (*NE).insert(b);
    }
  }
  updateAggregateBody();
}

void BHTree::printTree(int indent,std::ostream &os, std::string_view quadrant) {
  for (int i = 0; i < indent; ++i) {
    os << "  ";  // Two spaces per level of indentation
  }

  os << "Level "<< indent << " " << quadrant << " node:\n";

  for (int i = 0; i < indent + 1; ++i) {
    os << "  ";  // Two spaces per level of indentation
  }

  os << body << "\n";

  if (NW != nullptr && (*NW).body.mass > 0.0) {
    (*NW).printTree(indent + 1, os, "NW");
  }

  if (NE != nullptr && (*NE).body.mass > 0.0) {
    (*NE).printTree(indent + 1, os, "NE");
  }

  if (SW != nullptr && (*SW).body.mass > 0.0) {
    (*SW).printTree(indent + 1, os, "SW");
  }

  if (SE != nullptr && (*SE).body.mass > 0.0) {
    (*SE).printTree(indent + 1, os, "SE");
  }
}

void BHTree::updateVectorWithBodies(std::vector<Body>& bodies) {
  if (body.index != -1) {
    // Update the corresponding entry in the vector
    bodies[body.index] = body;
  }

  if (NW != nullptr) {
    (*NW).updateVectorWithBodies(bodies);
  }

  if (NE != nullptr) {
    (*NE).updateVectorWithBodies(bodies);
  }

  if (SW != nullptr) {
    (*SW).updateVectorWithBodies(bodies);
  }

  if (SE != nullptr) {
    (*SE).updateVectorWithBodies(bodies);
  }
}

Datavector BHTree::calculateForce(Body& b, double theta) {
    // If the node is an external node (and it is not body b), calculate the force exerted by the current node on b
    if (body.index != -1 && b != body) {
        return b.forceFrom(body);  // Use forceFrom function from Body class
    }

    // Otherwise, calculate the ratio s / d
    double s = quad.width;
    double d = b.position.minus(body.position).magnitude();

    // If s / d is less than a certain threshold (theta), treat this internal node as a single body
    if (s / d < theta) {
        return b.forceFrom(body);  // Use forceFrom function from Body class
    }

    // Otherwise, run the procedure recursively on each of the current node's children
    Datavector totalForce;

    if (NW != nullptr) {
        Datavector forceFromNW = (*NW).calculateForce(b,theta);
        totalForce = totalForce.plus(forceFromNW);
        //delete forceFromNW;
    }

    if (NE != nullptr) {
        Datavector forceFromNE = (*NE).calculateForce(b,theta);
        totalForce = totalForce.plus(forceFromNE);
        //delete forceFromNE;
    }

    if (SW != nullptr) {
        Datavector forceFromSW = (*SW).calculateForce(b,theta);
        totalForce = totalForce.plus(forceFromSW);
        //delete forceFromSW;
    }

    if (SE != nullptr) {
      Datavector forceFromSE = (*SE).calculateForce(b,theta);
        totalForce = totalForce.plus(forceFromSE);
        //delete forceFromSE;
    }

    return totalForce;
}
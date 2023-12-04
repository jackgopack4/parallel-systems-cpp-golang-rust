#include "bhtree.h"

BHTree::BHTree(Body& _body, Quad& _quad, BHTree& _NW, BHTree& _NE, BHTree& _SW, BHTree& _SE)
  : body(_body), quad(_quad), NW(&_NW), NE(&_NE), SW(&_SW), SE(&_SE) {}

BHTree::BHTree(Quad& _quad) : body(), quad(_quad), NW(nullptr), NE(nullptr), SW(nullptr), SE(nullptr) {}

BHTree::BHTree() : body(), quad(Quad(0.0, 0.0, 4.0)), NW(nullptr), NE(nullptr), SW(nullptr), SE(nullptr) {}
BHTree::~BHTree() {
  delete NW;
  delete NE;
  delete SW;
  delete SE;
}
void BHTree::insert(Body& b)
{
  // If the node is empty, insert the body here
  if (body.getMass() == 0.0) 
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
  double newWidth = quad.getWidth() / 2.0;
  double newHeight = quad.getHeight() / 2.0;
  double curMinX = quad.getMinX();
  double curMinY = quad.getMinY();
  Quad sw_quad(curMinX,curMinY,newWidth,newHeight);
  Quad nw_quad(curMinX,curMinY+newHeight,newWidth,newHeight);
  Quad se_quad(curMinX+newWidth,curMinY,newWidth,newHeight);
  Quad ne_quad(curMinX+newWidth,curMinY+newHeight,newWidth,newHeight);
  SW = new BHTree(sw_quad);
  NW = new BHTree(nw_quad);
  SE = new BHTree(se_quad);
  NE = new BHTree(ne_quad);
  //moveExistingBody();
  //insertIntoQuadrant(body);

}

void BHTree::updateAggregateBody()
{
  double totalMass = NW->body.getMass() + NE->body.getMass() + SW->body.getMass() + SE->body.getMass();

  if (totalMass > 0.0) {
    double weightedX = NW->body.getPosition().cartesian(0) * NW->body.getMass() + NE->body.getPosition().cartesian(0) * NE->body.getMass()
                      + SW->body.getPosition().cartesian(0) * SW->body.getMass() + SE->body.getPosition().cartesian(0) * SE->body.getMass();

    double weightedY = NW->body.getPosition().cartesian(1) * NW->body.getMass() + NE->body.getPosition().cartesian(1) * NE->body.getMass()
                      + SW->body.getPosition().cartesian(1) * SW->body.getMass() + SE->body.getPosition().cartesian(1) * SE->body.getMass();

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
    //body.getPosition().x = centerX;
    //body.getPosition().y = centerY;
    //body.setMass(totalMass);
}


// Helper function to insert a body into the appropriate quadrant
void BHTree::insertIntoQuadrant(Body& b) 
{
  double midX = quad.getMinX() + quad.getWidth() / 2.0;
  double midY = quad.getMinY() + quad.getHeight() / 2.0;

  if (b.getPosition().cartesian(0) <= midX) {
    if (b.getPosition().cartesian(1) <= midY) {
      SW->insert(b);
    } else {
      NW->insert(b);
    }
  } else {
    if (b.getPosition().cartesian(1) <= midY) {
      SE->insert(b);
    } else {
      NE->insert(b);
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

  if (NW != nullptr && NW->body.getMass() > 0.0) {
    NW->printTree(indent + 1, os, "NW");
  }

  if (NE != nullptr && NE->body.getMass() > 0.0) {
    NE->printTree(indent + 1, os, "NE");
  }

  if (SW != nullptr && SW->body.getMass() > 0.0) {
    SW->printTree(indent + 1, os, "SW");
  }

  if (SE != nullptr && SE->body.getMass() > 0.0) {
    SE->printTree(indent + 1, os, "SE");
  }
}

void BHTree::updateVectorWithBodies(std::vector<Body>& bodies) {
  if (body.getIndex() != -1) {
    // Update the corresponding entry in the vector
    bodies[body.getIndex()] = body;
  }

  if (NW != nullptr) {
    NW->updateVectorWithBodies(bodies);
  }

  if (NE != nullptr) {
    NE->updateVectorWithBodies(bodies);
  }

  if (SW != nullptr) {
    SW->updateVectorWithBodies(bodies);
  }

  if (SE != nullptr) {
    SE->updateVectorWithBodies(bodies);
  }
}

void forceFrom(double* x_comp, double* y_comp, Body& b0, Body& b1, double g) {
  auto dx = b0.position.cartesian(0)-b1.position.cartesian(0);
  auto dy = b0.position.cartesian(1)-b1.position.cartesian(1);
  double dist = sqrt(pow(dx,2)+pow(dy,2));
  double dist2 = dist;
  if (dist < b0.rlimit) dist2 = b0.rlimit;
  double Fx = (g * b1.mass*b0.mass*dx) / (dist*pow(dist2,2));
  double Fy = (g * b1.mass*b0.mass*dy) / (dist*pow(dist2,2));
  (*x_comp) = Fx;
  (*y_comp) = Fy;
}

void calculateForceHelper(BHTree* bht, Body& b0, Body& b1, double theta, Datavector* totalForce) {
  if (b0.mass <= 0.0000000001) return;
  double s = bht->quad.getWidth();
  auto b1_x = b1.position.cartesian(0);
  auto b1_y = b1.position.cartesian(1);
  auto b0_x = b0.position.cartesian(0);
  auto b0_y = b0.position.cartesian(1);
  auto diff_x = b1_x-b0_x;
  auto diff_y = b1_y-b0_y;
  double d = sqrt(pow(diff_x,2)+pow(diff_y,2));
  if(b1 == b0) return;
  if ((b0.getIndex() != -1) || s / d < theta) {
    double x_comp;
    double y_comp;
    forceFrom(&x_comp,&y_comp,b0,b1,b0.g);
    //Datavector* tmp_forcefrom = b1.forceFrom(&b0);
    //totalForce->plusEquals(tmp_forcefrom);
    totalForce->data[0]+=x_comp;
    totalForce->data[1]+=y_comp;
    //delete tmp_forcefrom;
    return;
  }
  if (bht->NW != nullptr) {
    calculateForceHelper(bht->NW,bht->NW->body,b1,theta,totalForce);
    calculateForceHelper(bht->NE,bht->NE->body,b1,theta,totalForce);
    calculateForceHelper(bht->SW,bht->SW->body,b1,theta,totalForce);
    calculateForceHelper(bht->SE,bht->SE->body,b1,theta,totalForce);
  }
}

Datavector* BHTree::calculateForce(Body& b, double theta) {
    Datavector* totalForce = new Datavector();
    calculateForceHelper(this,body,b,theta,totalForce);
    return totalForce;
}
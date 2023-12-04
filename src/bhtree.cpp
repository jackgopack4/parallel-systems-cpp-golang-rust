#include "bhtree.h"

BHTree::BHTree(Body& _body, Quad& _quad, BHTree& _NW, BHTree& _NE, BHTree& _SW, BHTree& _SE)
  : body(_body), quad(_quad), NW(&_NW), NE(&_NE), SW(&_SW), SE(&_SE) {}

BHTree::BHTree(Quad& _quad) : body(), quad(_quad), NW(nullptr), NE(nullptr), SW(nullptr), SE(nullptr) {}

BHTree::BHTree() : body(), quad(Quad(0.0, 0.0, 4.0)), NW(nullptr), NE(nullptr), SW(nullptr), SE(nullptr) {}
BHTree::~BHTree() {
  if(NW != nullptr) {
    delete NW;
    delete NE;
    delete SW;
    delete SE;
  }
}
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
  // check distance between point and body point, if too close, give up
  
  if (!body.checkAggregate()) {
    auto b0_x = body.position.data[0];
    auto b0_y = body.position.data[1];
    auto b1_x = b.position.data[0];
    auto b1_y = b.position.data[1];
    auto dist = sqrt(pow((b1_x-b0_x),2)+pow((b1_y-b0_y),2));
    if(dist < 0.0000000001) {
      body.mass += b.mass;
      return;
    }
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
  double totalMass = NW->body.mass + NE->body.mass + SW->body.mass + SE->body.mass;

  if (totalMass > 0.0) {
    double weightedX = NW->body.position.data[0] * NW->body.mass + NE->body.position.data[0] * NE->body.mass
                      + SW->body.position.data[0] * SW->body.mass + SE->body.position.data[0] * SE->body.mass;

    double weightedY = NW->body.position.data[1] * NW->body.mass + NE->body.position.data[1] * NE->body.mass
                      + SW->body.position.data[1] * SW->body.mass + SE->body.position.data[1] * SE->body.mass;

    double centerX = weightedX / totalMass;
    double centerY = weightedY / totalMass;

    // Update the aggregate body information in this node
    body.position.data[0] = centerX;
    body.position.data[1] = centerY;
    body.mass = totalMass;
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
  double midX = quad.minX + quad.width / 2.0;
  double midY = quad.minY + quad.height / 2.0;

  if (b.position.data[0] <= midX) {
    if (b.position.data[1] <= midY) {
      SW->insert(b);
    } else {
      NW->insert(b);
    }
  } else {
    if (b.position.data[1] <= midY) {
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

  if (NW != nullptr && NW->body.mass > 0.0) {
    NW->printTree(indent + 1, os, "NW");
  }

  if (NE != nullptr && NE->body.mass > 0.0) {
    NE->printTree(indent + 1, os, "NE");
  }

  if (SW != nullptr && SW->body.mass > 0.0) {
    SW->printTree(indent + 1, os, "SW");
  }

  if (SE != nullptr && SE->body.mass > 0.0) {
    SE->printTree(indent + 1, os, "SE");
  }
}

void BHTree::updateVectorWithBodies(std::vector<Body>& bodies) {
  if (body.index != -1) {
    // Update the corresponding entry in the vector
    bodies[body.index] = body;
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

void forceFrom(double* x_comp, double* y_comp, Body& b0, Body& b1) {
  auto dx = b0.position.data[0]-b1.position.data[0];
  auto dy = b0.position.data[1]-b1.position.data[1];
  double dist = sqrt(pow(dx,2)+pow(dy,2));
  double dist2 = dist;
  if (dist < b0.rlimit) dist2 = b0.rlimit;
  double Fx = (b0.g * b1.mass*b0.mass*dx) / (dist*pow(dist2,2));
  double Fy = (b0.g * b1.mass*b0.mass*dy) / (dist*pow(dist2,2));
  (*x_comp) += Fx;
  (*y_comp) += Fy;
}

void calculateForceHelper(BHTree* bht, Body& b1, double theta, double* x_comp, double* y_comp) {
  if(bht == nullptr) return;
  if (bht->body.mass <= 0.0000000001 || b1.mass <= 0.0000000001) return;
  double s = bht->quad.width;
  auto b1_x = b1.position.data[0];
  auto b1_y = b1.position.data[1];
  auto b0_x = bht->body.position.data[0];
  auto b0_y = bht->body.position.data[1];
  auto diff_x = b1_x-b0_x;
  auto diff_y = b1_y-b0_y;
  double d = sqrt(pow(diff_x,2)+pow(diff_y,2));
  if(b1 == bht->body) return;
  if ((bht->body.index != -1) || s / d < theta) {
    forceFrom(x_comp,y_comp,bht->body,b1);
    //Datavector* tmp_forcefrom = b1.forceFrom(&b0);
    //totalForce->plusEquals(tmp_forcefrom);
    //delete tmp_forcefrom;
    return;
  }
  if (bht->NW != nullptr) {
    if(bht->NW->body.mass > 0.0000000001) calculateForceHelper(bht->NW,b1,theta,x_comp,y_comp);
    if(bht->NE->body.mass > 0.0000000001) calculateForceHelper(bht->NE,b1,theta,x_comp,y_comp);
    if(bht->SW->body.mass > 0.0000000001) calculateForceHelper(bht->SW,b1,theta,x_comp,y_comp);
    if(bht->SE->body.mass > 0.0000000001) calculateForceHelper(bht->SE,b1,theta,x_comp,y_comp);
  }
}

void BHTree::calculateForce(Body& b, double theta, double* x_comp, double* y_comp) {
    //Datavector* totalForce = new Datavector();
    calculateForceHelper(this,b,theta, x_comp, y_comp);
    //return totalForce;
}
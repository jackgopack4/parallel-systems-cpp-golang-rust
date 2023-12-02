#include "bhtree.h"

BHTree::BHTree(Body& _body, Quad& _quad, BHTree& _NW, BHTree& _NE, BHTree& _SW, BHTree& _SE)
  : body(_body), quad(_quad), NW(&_NW), NE(&_NE), SW(&_SW), SE(&_SE) {}

BHTree::BHTree(Quad& _quad) : body(), quad(_quad), NW(nullptr), NE(nullptr), SW(nullptr), SE(nullptr) {}

BHTree::BHTree() : body(), quad(Quad(0.0, 0.0, 4.0)), NW(nullptr), NE(nullptr), SW(nullptr), SE(nullptr) {}

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
  SW = new BHTree(*(new Quad(curMinX,curMinY,newWidth,newHeight)));
  NW = new BHTree(*(new Quad(curMinX,curMinY+newHeight,newWidth,newHeight)));
  SE = new BHTree(*(new Quad(curMinX+newWidth,curMinY,newWidth,newHeight)));
  NE = new BHTree(*(new Quad(curMinX+newWidth,curMinY+newHeight,newWidth,newHeight)));

  //insertIntoQuadrant(body);

}

// Helper function to move the existing body to the appropriate quadrant and update aggregate information
void BHTree::moveExistingBody() {
    if (NW == nullptr) {
        // This is a leaf node, no need to move the existing body
        return;
    }
    insertIntoQuadrant(body);

    // Calculate the aggregate body information

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
        //isAggregate = true;
        //body(-1,{centerX,centerY},totalMass,true);
        //body.getPosition().x = centerX;
        //body.getPosition().y = centerY;
        //body.setMass(totalMass);
    }
}

// Helper function to insert a body into the appropriate quadrant
void BHTree::insertIntoQuadrant(Body& b) 
{
  double midX = quad.getMinX() + quad.getWidth() / 2.0;
  double midY = quad.getMinY() + quad.getWidth() / 2.0;

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
}

void BHTree::printTree(int indent,std::ostream &os) {
    for (int i = 0; i < indent; ++i) {
        os << "  ";  // Two spaces per level of indentation
    }

    os << "Level "<< indent << " Node Contents:\n";

    for (int i = 0; i < indent + 1; ++i) {
        os << "  ";  // Two spaces per level of indentation
    }

    os << body << "\n";

    if (NW != nullptr && NW->body.getMass() > 0.0) {
        NW->printTree(indent + 1, os);
    }

    if (NE != nullptr && NE->body.getMass() > 0.0) {
        NE->printTree(indent + 1, os);
    }

    if (SW != nullptr && SW->body.getMass() > 0.0) {
        SW->printTree(indent + 1, os);
    }

    if (SE != nullptr && SE->body.getMass() > 0.0) {
        SE->printTree(indent + 1, os);
    }
}
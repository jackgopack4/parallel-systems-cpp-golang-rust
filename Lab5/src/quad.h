#ifndef QUAD_H
#define QUAD_H

#include "body.h"  

class Quad {
private:

public:
  double minX;
  double minY;
  double width;
  double height;
  // Constructor
  Quad(double _minX, double _minY, double _width);
  Quad(double _minX, double _minY, double _width, double _height);

  // Getter functions
  double getMinX() const;
  double getMinY() const;
  double getWidth() const;
  double getHeight() const;
  bool isSquare() const;

  // Function to check if a Body is inside the Quad
  bool contains(Body& body) const;
};

#endif // QUAD_H

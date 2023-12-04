#include <iostream>
#include "quad.h"

Quad::Quad(double _minX, double _minY, double _width)
  : minX(_minX), minY(_minY), width(_width), height(_width) {}

Quad::Quad(double _minX, double _minY, double _width, double _height)
  : minX(_minX), minY(_minY), width(_width), height(_height) {}

// Getter functions
double Quad::getMinX() const 
{
  return minX;
}

double Quad::getMinY() const 
{
  return minY;
}

double Quad::getWidth() const 
{
  return width;
}

double Quad::getHeight() const 
{
  return height;
}

// Function to check if a Body is inside the Quad
bool Quad::contains(Body& body) const {
  //Datavector bodyPos = body.getPosition();
  return (body.position_arr[0] >= minX && body.position_arr[0] <= minX + width && body.position_arr[1] >= minY && body.position_arr[1] <= minY + width);
}
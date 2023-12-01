#include <vector>
#include <iostream>

#include "datavector.h"

#ifndef BODY_H
#define BODY_H

class Body 
{
  private:
    Datavector position;
    Datavector velocity;
    double mass;
    double g;
    double rlimit;
  
  public:
    Body(const Datavector& initialPosition, const Datavector& initialVelocity, double initialMass, double gravity, double limit);
    Body(Datavector _position, Datavector _velocity, double _mass);
    //Body(Datavector _position, Datavector _velocity, double _mass, double _g, double _rlimit);
    void move(Datavector* force, double dt);
    Datavector* forceFrom(Body* b);
    friend std::ostream& operator << (std::ostream &os, Body &b)
    {
      return (os << "Body - position:  " << b.position << ", velocity: " << b.velocity << ", mass " << b.mass);
    }
};

#endif
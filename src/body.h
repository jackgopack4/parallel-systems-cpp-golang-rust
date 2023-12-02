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
    int index;
    bool isAggregate;
  
  public:
    Body(const Body& other);
    Body();
    Body(int _index, Datavector& initialPosition, Datavector& initialVelocity, double initialMass, double gravity, double limit);
    Body(int _index, Datavector _position, Datavector _velocity, double _mass);
    Body(int _index, Datavector _position, Datavector _velocity, double _mass, bool _aggregate);
    //Body(Datavector _position, Datavector _velocity, double _mass, double _g, double _rlimit);
    void move(Datavector* force, double dt);
    Datavector* forceFrom(Body* b);
    Datavector& getPosition();
    Datavector& getVelocity();
    double getMass();
    int getIndex();
    void setMass(double _mass);
    void setPosition(std::vector<double> _position);
    void makeAggregate();
    bool operator==(Body& other);
    bool operator!=(Body& other);
    friend std::ostream& operator << (std::ostream &os, Body &b)
    {
      if (b.isAggregate) 
      {
        os << "Aggregate - ";
      } else 
      {
        os << "Body["<< b.index << "] - ";
      }
      os << "position:  " << b.position << ", velocity: " << b.velocity << ", mass " << b.mass;
      return os;
    }
};

#endif
#include <vector>
#include <iostream>
#include <cmath>

//#include "datavector.h"

#ifndef BODY_H
#define BODY_H

class Body 
{
  private:
  
  public:
    double position_arr[2]{0.0,0.0};
    double velocity_arr[2]{0.0,0.0};
    double mass;
    double g;
    double rlimit;
    int index;
    bool isAggregate;

    Body(const Body& other);
    Body();
    //~Body();
    Body(int _index, double* _position, double* _velocity, double _mass);
    Body(int _index, double* _position, double* _velocity, double _mass, bool _aggregate);

    void move(double* force, double dt);
    void forceFromArr(Body& b0, Body& b1, double* force_arr);
    double getMass();
    int getIndex();
    void setMass(double _mass);
    void setPosition(double* _position);
    void makeAggregate();
    bool checkAggregate();
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
      os << "position " << b.position_arr << ", velocity " << b.velocity_arr << ", mass " << b.mass;
      return os;
    }
};

void positionAndVelocityUpdate(Body* b, double* f, double dt);
#endif
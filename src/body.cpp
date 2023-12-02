#include "body.h"

Body::Body(const Body& other) : position(other.position), velocity(other.velocity), mass(other.mass), index(other.index) {
  g = 0.0001;
  rlimit = 0.03;
}

Body::Body():
  position(Datavector({0.0,0.0})),
  velocity(Datavector({0.0,0.0}))
{
  mass = -1.0;
  index = -1;
  g = 0.0001;
  rlimit = 0.03;
}

Body::Body(int _index, Datavector _position, Datavector _velocity, double _mass):
  position(_position),
  velocity(_velocity),
  mass(_mass),
  index(_index)
{
  g = 0.0001;
  rlimit = 0.03;
}

Body::Body(int _index, Datavector& initialPosition, Datavector& initialVelocity, double initialMass, double gravity, double limit)
        : position(initialPosition), velocity(initialVelocity), mass(initialMass), g(gravity), rlimit(limit), index(_index) {
    }

void Body::move(Datavector* force, double dt)
{
  Datavector* a = force->scale(1/mass);
  velocity = *(velocity.plus(a->scale(dt)));
  position = *(position.plus(velocity.scale(dt)));
}

Datavector* Body::forceFrom(Body* b)
{
  //Body* a = this;
  Datavector* delta = b->position.minus(&(this->position));
  double dist = delta->magnitude();
  if (dist < rlimit)
  {
    dist = rlimit;
  }
  double magnitude = (g * this->mass * b->mass) / (dist * dist);
  return delta->direction()->scale(magnitude);
}

Datavector& Body::getPosition() 
{
  return position;
}

int Body::getIndex()
{
  return index;
}
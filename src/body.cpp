#include "body.h"

Body::Body(Datavector _position, Datavector _velocity, double _mass):
  position(_position),
  velocity(_velocity),
  mass(_mass)
{
  g = 0.0001;
  rlimit = 0.03;
}

Body::Body(const Datavector& initialPosition, const Datavector& initialVelocity, double initialMass, double gravity, double limit)
        : position(initialPosition), velocity(initialVelocity), mass(initialMass), g(gravity), rlimit(limit) {
    }

void Body::move(Datavector* force, double dt)
{
  Datavector* a = force->scale(1/mass);
  velocity = *(velocity.plus(a->scale(dt)));
  position = *(position.plus(velocity.scale(dt)));
}

Datavector* Body::forceFrom(Body* b)
{
  Body* a = this;
  Datavector* delta = b->position.minus(&(a->position));
  double dist = delta->magnitude();
  if (dist < rlimit)
  {
    dist = rlimit;
  }
  double magnitude = (g * a->mass * b->mass) / (dist * dist);
  return delta->direction()->scale(magnitude);
}
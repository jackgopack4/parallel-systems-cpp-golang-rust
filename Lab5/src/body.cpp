#include "body.h"

Body::Body():
  position(Datavector({0.0,0.0})),
  velocity(Datavector({0.0,0.0})),
  mass(0.0),
  g(0.0001),
  rlimit(0.03),
  index(-1),
  isAggregate(false)
{}

Body::Body(int _index, Datavector _position, Datavector _velocity, double _mass):
  position(_position),
  velocity(_velocity),
  mass(_mass),
  g(0.0001),
  rlimit(0.03),
  index(_index),
  isAggregate(false)
{}

Body::Body(int _index, Datavector _position, Datavector _velocity, double _mass, bool _aggregate):
  position(_position),
  velocity(_velocity),
  mass(_mass),
  index(_index),
  isAggregate(_aggregate)
{
  g = 0.0001;
  rlimit = 0.03;
}

Body::Body(int _index, Datavector& initialPosition, Datavector& initialVelocity, double initialMass, double gravity, double limit)
        : position(initialPosition), velocity(initialVelocity), mass(initialMass), g(gravity), rlimit(limit), index(_index) {
    }

void Body::move(Datavector* force, double dt)
{
  double ax = force->data[0] / mass;
  double ay = force->data[1] / mass;

  double Vxdt = velocity.data[0]*dt;
  double Vydt = velocity.data[1]*dt;

  double half_ax_dt2 = 0.5*ax*pow(dt,2);
  double half_ay_dt2 = 0.5*ay*pow(dt,2);

  double axdt = ax*dt;
  double aydt = ay*dt;

  double px_prime = position.data[0] + Vxdt + half_ax_dt2;
  double py_prime = position.data[1] + Vydt + half_ay_dt2;

  double vx_prime = velocity.data[0]+axdt;
  double vy_prime = velocity.data[1]+aydt;

  position.data[0] = px_prime;
  position.data[1] = py_prime;

  velocity.data[0] = vx_prime;
  velocity.data[1] = vy_prime;

  //velocity = *(velocity.plus(a->scale(dt)));
  //position = *(position.plus(velocity.scale(dt)));
  if(position.data[0] <= 0.0 || position.data[0] >= 4.0
    || position.data[1] <= 0.0 || position.data[1] >= 4.0) 
  {
      mass = -1.0;
  }
}

Datavector* Body::forceFrom(Body* b)
{
  //Body* a = this;
  Datavector* delta = b->position.minus(&(this->position));
  
  double dist = delta->magnitude();
  double dx = delta->data[0];
  double dy = delta->data[1];
  delete delta;
  double dist2 = dist;
  if (dist < rlimit)
  {
    dist2 = rlimit;
  }
  double Fx = (g * this->mass * b->mass * dx) / (dist*dist2*dist2);
  double Fy = (g * this->mass * b->mass * dy) / (dist*dist2*dist2);

  return new Datavector({Fx,Fy});
  //double magnitude = (g * this->mass * b->mass) / (dist * dist);
  //return delta->direction()->scale(magnitude);
}

Datavector& Body::getPosition() 
{
  return position;
}

Datavector& Body::getVelocity()
{
  return velocity;
}

int Body::getIndex()
{
  return index;
}

double Body::getMass()
{
  return mass;
}

void Body::setMass(double _mass)
{
  mass = _mass;
}

void Body::setPosition(std::vector<double> _position)
{
  Datavector* tmp_pos = new Datavector(_position);
  position = *(tmp_pos);
  delete tmp_pos;
  //position(Datavector(_position));
}

void Body::makeAggregate()
{
  isAggregate = true;
  index = -1;
}

bool Body::operator==(Body& other)
{
  return position == other.position && index == other.index;
}

bool Body::operator!=(Body& other)
{
  return !(*this == other);
}

bool Body::checkAggregate()
{
  return isAggregate;
}
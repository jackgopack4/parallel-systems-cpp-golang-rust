#include "body.h"

Body::Body(const Body& other) : position(other.position), velocity(other.velocity), mass(other.mass), index(other.index) {
  g = 0.0001;
  rlimit = 0.03;
  isAggregate = false;
}

Body::Body():
  position(Datavector({0.0,0.0})),
  velocity(Datavector({0.0,0.0}))
{
  mass = 0.0;
  index = -1;
  g = 0.0001;
  rlimit = 0.03;
  isAggregate = false;
}

Body::~Body()
{
  position.clear();
  velocity.clear();
}

Body::Body(int _index, Datavector _position, Datavector _velocity, double _mass):
  position(_position),
  velocity(_velocity),
  mass(_mass),
  index(_index)
{
  g = 0.0001;
  rlimit = 0.03;
  isAggregate = false;
}

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

void Body::move(Datavector& force, double dt)
{
  Datavector a = force.scale(1/mass);
  //double ax = a->cartesian(0);
  //double ay = a->cartesian(1);

  Datavector P_term2 = velocity.scale(dt);
  double Vxdt = P_term2.cartesian(0);
  double Vydt = P_term2.cartesian(1);

  Datavector P_term3 = a.scale(0.5*dt*dt);
  double half_ax_dt2 = P_term3.cartesian(0);
  double half_ay_dt2 = P_term3.cartesian(1);

  Datavector a_dt = a.scale(dt);
  double axdt = a_dt.cartesian(0);
  double aydt = a_dt.cartesian(1);

  double px_prime = position.cartesian(0) + Vxdt + half_ax_dt2;
  double py_prime = position.cartesian(1) + Vydt + half_ay_dt2;

  double vx_prime = velocity.cartesian(0)+axdt;
  double vy_prime = velocity.cartesian(1)+aydt;
  //delete &position;
  //delete &velocity;
  position = Datavector({px_prime,py_prime});
  velocity = Datavector({vx_prime,vy_prime});
  //delete a;
  //delete P_term2;
  //delete P_term3;
  //delete a_dt;
  //velocity = *(velocity.plus(a->scale(dt)));
  //position = *(position.plus(velocity.scale(dt)));
  if(position.cartesian(0) <= 0.0 || position.cartesian(0) >= 4.0
    || position.cartesian(1) <= 0.0 || position.cartesian(1) >= 4.0) 
  {
      mass = -1.0;
  }
}

Datavector Body::forceFrom(Body b)
{
  //Body* a = this;
  Datavector delta = position.minus(b.position);
  
  double dist = delta.magnitude();
  double dx = delta.cartesian(0);
  double dy = delta.cartesian(1);
  double dist2 = dist;
  if (dist < rlimit)
  {
    dist2 = rlimit;
  }
  double Fx = (g * mass * b.mass * dx) / (dist*dist2*dist2);
  double Fy = (g * mass * b.mass * dy) / (dist*dist2*dist2);

  Datavector res({Fx,Fy});
  return res;
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
  //delete &position;
  position = _position;
  //position(Datavector(_position));
}

void Body::makeAggregate()
{
  isAggregate = true;
  index = -1;
}

bool Body::operator==(Body& other)
{
  return position == other.getPosition() && mass == other.getMass();
}

bool Body::operator!=(Body& other)
{
  return !(*this == other);
}

bool Body::checkAggregate()
{
  return isAggregate;
}
#include "body.h"

void positionAndVelocityUpdate(Body* b, double* force, double dt) {
  double ax = force[0]*(1/b->mass);
  double ay = force[1]*(1/b->mass);

  double Vxdt = b->velocity_arr[0]*dt;
  double Vydt = b->velocity_arr[1]*dt;

  double half_ax_dt2 = ax*0.5*pow(dt,2);
  double half_ay_dt2 = ay*0.5*pow(dt,2);

  double axdt = ax*dt;
  double aydt = ay*dt;

  double px_prime = b->position_arr[0] + Vxdt + half_ax_dt2;
  double py_prime = b->position_arr[1] + Vydt + half_ay_dt2;

  double vx_prime = b->velocity_arr[0] + axdt;
  double vy_prime = b->velocity_arr[1] + aydt;

  b->position_arr[0] = px_prime;
  b->position_arr[1] = py_prime;
  b->velocity_arr[0] = vx_prime;
  b->velocity_arr[1] = vy_prime;

  if(b->position_arr[0] <= 0.0 || b->position_arr[0] >= 4.0
    || b->position_arr[1] <= 0.0 || b->position_arr[1] >= 4.0) 
  {
    b->mass = -1.0;
  }
}

Body::Body(const Body& other) : index(other.index) {
  g = 0.0001;
  rlimit = 0.03;
  isAggregate = false;
  position_arr[0] = other.position_arr[0];
  position_arr[1] = other.position_arr[1];
  velocity_arr[0] = other.velocity_arr[0];
  velocity_arr[1] = other.velocity_arr[1];
}

Body::Body()
{
  mass = 0.0;
  index = -1;
  g = 0.0001;
  rlimit = 0.03;
  isAggregate = false;
}
/*
Body::~Body()
{
  position = nullptr;
  velocity = nullptr;
}
*/

Body::Body(int _index, double* _position, double* _velocity, double _mass)
        : mass(_mass), index(_index) {
  position_arr[0] = _position[0];
  position_arr[1] = _position[1];
  velocity_arr[0] = _velocity[0];
  velocity_arr[1] = _velocity[1];
  g = 0.0001;
  rlimit = 0.03;
  isAggregate = false;
}
Body::Body(int _index, double* _position, double* _velocity, double _mass, bool _aggregate)
        : mass(_mass), index(_index), isAggregate(_aggregate) {
  position_arr[0] = _position[0];
  position_arr[1] = _position[1];
  velocity_arr[0] = _velocity[0];
  velocity_arr[1] = _velocity[1];
  g = 0.0001;
  rlimit = 0.03;
}
void Body::move(double* force, double dt)
{
  double ax = force[0]*(1/mass);
  double ay = force[1]*(1/mass);

  double Vxdt = velocity_arr[0]*dt;
  double Vydt = velocity_arr[1]*dt;

  double half_ax_dt2 = ax*0.5*pow(dt,2);
  double half_ay_dt2 = ay*0.5*pow(dt,2);

  double axdt = ax*dt;
  double aydt = ay*dt;

  double px_prime = position_arr[0] + Vxdt + half_ax_dt2;
  double py_prime = position_arr[1] + Vydt + half_ay_dt2;

  double vx_prime = velocity_arr[0] + axdt;
  double vy_prime = velocity_arr[1] + aydt;

  position_arr[0] = px_prime;
  position_arr[1] = py_prime;
  velocity_arr[0] = vx_prime;
  velocity_arr[1] = vy_prime;

  if(position_arr[0] <= 0.0 || position_arr[0] >= 4.0
    || position_arr[1] <= 0.0 || position_arr[1] >= 4.0) 
  {
    mass = -1.0;
  }
}

void Body::forceFromArr(Body& b0, Body& b1, double* force_arr)
{ // calculates force from b1 applied to b0
  if (b0.mass == 0 || b1.mass == 0) {
    return;
  }
  //std::cout << "force on body["<<b0.index<<"] before body["<<b1.index<<"] acts:\n";
  //std::cout << "[" << force_arr[0] << ", " << force_arr[1] << "]\n";
  double dx = b1.position_arr[0] - b0.position_arr[0];
  double dy = b1.position_arr[1] - b0.position_arr[1];
  double dist = sqrt(pow(dx,2)+pow(dy,2));
  double dist2 = dist;
  if (dist < rlimit) dist2 = rlimit;
  double Fx = (g*b0.mass*b1.mass*dx) / (dist*pow(dist2,2));
  double Fy = (g*b0.mass*b1.mass*dy) / (dist*pow(dist2,2));

  force_arr[0] += Fx;
  force_arr[1] += Fy;

  //double res[2]{Fx,Fy};
  //return res;
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

void Body::setPosition(double* _position)
{
  position_arr[0] = _position[0];
  position_arr[1] = _position[1];
}

void Body::makeAggregate()
{
  isAggregate = true;
  index = -1;
}

bool Body::checkAggregate()
{
  return isAggregate;
}

bool Body::operator==(Body& other)
{
  return index == other.index 
    && position_arr[0] == other.position_arr[0] 
    && position_arr[1] == other.position_arr[1]
    && mass == other.mass;
}

bool Body::operator!=(Body& other)
{
  return !(*this == other);
}
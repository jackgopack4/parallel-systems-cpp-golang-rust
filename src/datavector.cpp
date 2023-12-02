#include "datavector.h"

Datavector::Datavector(const Datavector& other) : n(other.n), data(other.data) {}

Datavector::Datavector(int _n) 
{
  n = _n;
  data.reserve(_n);
}

Datavector::Datavector(std::vector<double> _data) 
{
  n = _data.size();
  data.reserve(n);
  for (double d: _data) {
    data.push_back(d);
  }
}

Datavector::Datavector(int _n, double _data[])
{
  n = _n;
  for (auto i=0;i<n;++i) {
    data.push_back(_data[i]);
  }
}

int Datavector::size() 
{
  return n;
}

double Datavector::dot(Datavector* other) 
{
  // not adding error checking, vectors all same size for this function
  double sum{0.0};
  for(auto i=0;i<n;++i) 
  {
    sum += data[i] * other->data[i];
  }
  return sum;
}

double Datavector::magnitude() 
{
  return std::sqrt(this->dot(this));
}

double Datavector::distanceTo(Datavector* other) 
{
  return this->minus(other)->magnitude();
}

Datavector* Datavector::plus(Datavector* other) 
{
  Datavector* sum = new Datavector(n);
  for(auto i=0;i<n;++i) 
  {
    sum->data[i] = data[i] + other->data[i];
  }
  return sum;
}

Datavector* Datavector::minus(Datavector* other) 
{
  Datavector* diff = new Datavector(n);
  for(auto i=0; i<n; ++i) 
  {
    diff->data[i] = data[i] + other->data[i];
  }
  return diff;
}

double Datavector::cartesian(int i) 
{
  return data[i];
}

Datavector* Datavector::scale(double factor) 
{
  Datavector* scaled = new Datavector(n);
  for (auto i=0;i<n;++i) 
  {
    scaled->data[i] = factor * data[i];
  }
  return scaled;
}

Datavector* Datavector::direction() 
{
  return this->scale(1.0 / this->magnitude());
}

bool Datavector::operator==(Datavector& other)
{
  if (size() != other.size()) return false;
  for (int i{0}; i< size(); ++i)
  {
    if (cartesian(i) != other.cartesian(i)) return false;
  }
  return true;
}
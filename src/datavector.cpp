#include "datavector.h"

Datavector::Datavector(const Datavector& other) : n(other.n), data(other.data) {}

//Datavector::Datavector(Datavector& other) : n(other.n), data(other.data) {}

Datavector::Datavector() : n(2), data({0.0,0.0}) {}

Datavector::~Datavector() 
{
  data.clear();
}

void Datavector::clear() {
  data.clear();
}

Datavector::Datavector(int _n) 
{
  n = _n;
  data.reserve(_n);
  for(auto i=0;i<n;++i) {
    data.push_back(0.0);
  }
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
  data.reserve(n);
  for (auto i=0;i<n;++i) {
    data.push_back(_data[i]);
  }
}

int Datavector::size() 
{
  return n;
}

double Datavector::dot(Datavector& other) 
{
  // not adding error checking, vectors all same size for this function
  double sum{0.0};
  for(auto i=0;i<n;++i) 
  {
    sum += (data[i] * other.data[i]);
  }
  return sum;
}

double Datavector::magnitude() 
{
  double sum{0.0};
  for(auto i=0;i<n;++i)
  {
    sum += data[i] * data[i];
  }
  return sum;
}

double Datavector::distanceTo(Datavector& other) 
{
  return minus(other).magnitude();
}

Datavector Datavector::plus(Datavector& other) 
{
  std::vector<double> sums(n);
  for(auto i=0;i<n;++i) 
  {
    sums[i] = data[i] + other.cartesian(i);
  }
  Datavector sum = Datavector(sums);
  return sum;
}

Datavector Datavector::minus(Datavector& other) 
{
  //Datavector diff(n);
  std::vector<double> diffs(n);
  for(auto i=0; i<n; ++i) 
  {
    diffs[i] = data[i] - other.cartesian(i);
  }
  Datavector diff(diffs);
  return diff;
}

double Datavector::cartesian(int i) 
{
  return data[i];
}

Datavector Datavector::scale(double factor) 
{
  std::vector<double> scales(n);
  for (auto i=0;i<n;++i) 
  {
    scales[i] = factor * data[i];
  }

  Datavector scaled(scales);
  return scaled;
}

Datavector Datavector::direction() 
{
  return scale(1.0 / magnitude());
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
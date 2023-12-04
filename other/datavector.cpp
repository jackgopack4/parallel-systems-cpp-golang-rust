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

Datavector::Datavector(int _n): n(_n)
{
  std::vector<double> vector2(_n, 0.0);
  data = vector2;
}

Datavector::Datavector(std::vector<double> _data): n((int) _data.size()), data(_data) {}

Datavector::Datavector(int _n, double _data[]): n(_n), data(_data,_data+_n){}

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
  return sqrt(sum);
}

double Datavector::distanceTo(Datavector& other) 
{
  Datavector tmp = minus(other);
  return tmp.magnitude();
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
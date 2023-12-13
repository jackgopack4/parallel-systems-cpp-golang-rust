#include <vector>
#include <iostream>
#include <cmath>

#ifndef DATAVECTOR_H
#define DATAVECTOR_H

class Datavector
{
  private:
  public:
    int n{};
    std::vector<double> data;
    Datavector(const Datavector& other);
    //Datavector(Datavector& other);
    Datavector(Datavector* other);
    //Datavector(Datavector& _d);
    Datavector(int _n);
    Datavector(std::vector<double> _data);
    Datavector(int _n, double _data[]);
    Datavector();
    ~Datavector();
    void clear();
    int size();
    void plusEquals(Datavector* other);
    double dot(Datavector& other);
    double magnitude();
    double distanceTo(Datavector& other);
    Datavector plus(Datavector& other);
    Datavector minus(Datavector& other);
    double cartesian(int i);
    Datavector scale(double factor);
    Datavector direction();
    // Overload the equality operator
    bool operator==(Datavector& other);
    friend std::ostream& operator << (std::ostream &os, Datavector &v)
    {
      return (os << "Vector magnitude: " << v.magnitude() << ", x-comp: " << v.data[0] << ", y-comp: " << v.data[1]);
    }
    friend std::ostream& operator << (std::ostream &os, Datavector* v)
    {
      return (os << "Vector magnitude: " << v->magnitude() << ", x-comp: " << v->data[0] << ", y-comp: " << v->data[1]);
    }

};

#endif
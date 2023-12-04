#include <fstream>
#include <sstream>
#include <iostream>

#include "bodyfilereader.h"

BodyFileReader::BodyFileReader(const std::string& filename) : filename(filename) {}

std::vector<Body> BodyFileReader::readBodies() {
    std::vector<Body> bodies;


    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return bodies;
    }

    int numBodies;
    if (!(file >> numBodies)) {
        std::cerr << "Error: Failed to read the number of bodies from " << filename << std::endl;
        return bodies;
    }
    //bodies.resize(numBodies);
    //std::cout << "resized bodies\n";
    for (int i = 0; i < numBodies; ++i) {
        int index;
        double posX, posY, mass, velX, velY;

        if (!(file >> index >> posX >> posY >> mass >> velX >> velY)) {
            std::cerr << "Error: Failed to read body information from line " << i + 2 << " of " << filename << std::endl;
            return bodies;
        }
        std::cout << "mass read as: " << mass << std::endl;
        //Datavector position({posX, posY});
        //Datavector velocity({velX, velY});
        double* position_arr = new double[2]{posX,posY};
        double* velocity_arr = new double[2]{velX,velY};
        Body body(index, position_arr, velocity_arr, mass);
        bodies.push_back(body);
        delete position_arr;
        delete velocity_arr;
    }

    return bodies;
}

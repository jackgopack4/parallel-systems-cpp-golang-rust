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
    bodies.resize(numBodies);
    //std::cout << "resized bodies\n";
    for (int i = 0; i < numBodies; ++i) {
        int index;
        double posX, posY, mass, velX, velY;

        if (!(file >> index >> posX >> posY >> mass >> velX >> velY)) {
            std::cerr << "Error: Failed to read body information from line " << i + 2 << " of " << filename << std::endl;
            return bodies;
        }

        Datavector position({posX, posY});
        Datavector velocity({velX, velY});
        Body body(index,position, velocity, mass);
        bodies[index] = body;
    }

    return bodies;
}

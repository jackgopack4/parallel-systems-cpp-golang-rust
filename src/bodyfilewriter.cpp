#include "bodyfilewriter.h"
#include <fstream>
#include <iostream>

BodyFileWriter::BodyFileWriter(const std::string& filename) : filename(filename) {}

bool BodyFileWriter::writeBodies(std::vector<Body>& bodies) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return false;
    }

    file << bodies.size() << std::endl; // Write the number of bodies in the first line

    for (Body& body : bodies) {
        //Datavector& pos = body.getPosition();
        //Datavector& vel = body.getVelocity();

        //double 

        file << body.getIndex() << " "
             << body.position_arr[0] << " " << body.position_arr[1] << " "
             << body.getMass() << " "
             << body.velocity_arr[0] << " " << body.velocity_arr[1] << std::endl;
    }

    return true;
}

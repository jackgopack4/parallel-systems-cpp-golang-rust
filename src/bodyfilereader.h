#ifndef BODY_FILEREADER_H
#define BODY_FILEREADER_H

#include <vector>
#include "body.h"

class BodyFileReader {
public:
    // Constructor
    BodyFileReader(const std::string& filename);

    // Function to read the file and create a vector of Body items
    std::vector<Body> readBodies();

private:
    std::string filename;
};

#endif // BODY_FILEREADER_H
#ifndef BODY_FILEWRITER_H
#define BODY_FILEWRITER_H

#include <vector>
#include "body.h"  // Assuming you have a Body class definition

class BodyFileWriter {
public:
    // Constructor
    BodyFileWriter(const std::string& filename);

    // Function to write the list of Body items to file
    bool writeBodies(std::vector<Body>& bodies);

private:
    std::string filename;
};

#endif // BODY_FILEWRITER_H

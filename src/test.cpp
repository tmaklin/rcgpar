#include "rcg.hpp"

#include <iostream>

#include "version.h"

int main() {
    std::cerr << "rcg-MPI-" << RCGMPI_BUILD_VERSION << std::endl;
    return 0;
}

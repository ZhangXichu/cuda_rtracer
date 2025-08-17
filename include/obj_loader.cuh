#ifndef OBJ_LOADER_HPP
#define OBJ_LOADER_HPP

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <limits>
#include <vector_types.h> 
#include <stdexcept>

class ObjLoader {
public:
    bool load(const std::string& filename);
    void normalize();
    void upload_to_device(float*& d_vertices, int& vcount,
                          int3*& d_faces, int& fcount) const;

private:
    std::vector<float> _vertices; // x,y,z triples
    std::vector<int3> _faces;     // each face is a triangle (vertex indices)
};

#endif // OBJ_LOADER_HPP